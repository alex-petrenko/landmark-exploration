import random
import sys
from collections import deque

import numpy as np

from algorithms.algo_utils import main_observation, num_env_steps
from algorithms.buffer import Buffer
from algorithms.multi_env import MultiEnv
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
from algorithms.trajectory import TrajectoryBuffer, Trajectory
from utils.envs.envs import create_env
from utils.timing import Timing
from utils.utils import log


def generate_training_data(trajectories, params):
    timing = Timing()
    with timing.timeit('trajectories'):
        close, far = params.reachable_threshold, params.unreachable_threshold

        trajectory_joined = Trajectory(0)

        trajectory_idx = []
        episode_ends = []
        for i, t in enumerate(trajectories):
            trajectory_joined.add_trajectory(t)
            trajectory_idx.extend([i] * len(t))
            episode_ends.append(len(trajectory_joined))

        obs = trajectory_joined.obs

        indices = list(range(len(trajectory_joined)))
        np.random.shuffle(indices)

        buffer = Buffer()
        num_close, num_far = 0, 0

        for i in indices:
            # sample close observation pair
            close_i = min(i + close, len(trajectory_joined))
            first_idx = i
            second_idx = np.random.randint(i, close_i)

            if trajectory_idx[first_idx] == trajectory_idx[second_idx]:
                if params.reachability_symmetric and random.random() < 0.5:
                    first_idx, second_idx = second_idx, first_idx

                buffer.add(obs_first=obs[first_idx], obs_second=obs[second_idx], labels=0)
                num_close += 1

            # sample far observation pair
            next_episode_end = 0
            for next_episode_end in episode_ends:
                if next_episode_end > i:
                    break

            if random.random() < 0.3:
                max_len = len(trajectory_joined)
            else:
                max_len = next_episode_end

            far_i = min(i + far, max_len)

            if far_i < max_len:
                first_idx = i
                second_idx = np.random.randint(far_i, max_len)
                if params.reachability_symmetric and random.random() < 0.5:
                    first_idx, second_idx = second_idx, first_idx

                buffer.add(obs_first=obs[first_idx], obs_second=obs[second_idx], labels=1)
                num_far += 1

    log.info(
        'Processed %d trajectories, total %d, close %d, far %d, timing: %s',
        len(trajectories), len(buffer), num_close, num_far, timing,
    )

    return buffer


def train_distance_net(agent, trajectories, params, env_steps):
    buffer = generate_training_data(trajectories, params)

    reachability = agent.curiosity.reachability
    reach_step = agent.curiosity.step.eval(session=agent.session)

    summary = None
    prev_loss = 1e10
    num_epochs = params.reachability_train_epochs
    batch_size = params.reachability_batch_size

    log.info('Training reachability %d pairs, batch %d, epochs %d', len(buffer), batch_size, num_epochs)

    for epoch in range(num_epochs):
        losses = []
        buffer.shuffle_data()
        obs_first, obs_second, labels = buffer.obs_first, buffer.obs_second, buffer.labels

        for i in range(0, len(obs_first) - 1, batch_size):
            # noinspection PyProtectedMember
            with_summaries = agent._should_write_summaries(reach_step) and summary is None
            summaries = [agent.curiosity.summaries] if with_summaries else []

            start, end = i, i + batch_size

            result = agent.session.run(
                [reachability.loss, agent.curiosity.train_reachability] + summaries,
                feed_dict={
                    reachability.ph_obs_first: obs_first[start:end],
                    reachability.ph_obs_second: obs_second[start:end],
                    reachability.ph_labels: labels[start:end],
                }
            )

            reach_step += 1
            # noinspection PyProtectedMember
            agent._maybe_save(reach_step, env_steps)
            losses.append(result[0])

            if with_summaries:
                summary = result[-1]
                agent.summary_writer.add_summary(summary, global_step=env_steps)

        # check loss improvement at the end of each epoch, early stop if necessary
        avg_loss = np.mean(losses)
        if avg_loss >= prev_loss:
            log.info('Early stopping after %d epochs because reachability did not improve', epoch + 1)
            log.info('Was %.4f now %.4f, ratio %.3f', prev_loss, avg_loss, avg_loss / prev_loss)
            break
        prev_loss = avg_loss

    return reach_step


def train_loop(agent, multi_env):
    params = agent.params

    observations = main_observation(multi_env.reset())

    trajectory_buffer = TrajectoryBuffer(multi_env.num_envs)

    num_steps = 0
    training_steps = 0

    loop_time = deque([], maxlen=1000)
    advanced_steps = deque([], maxlen=1000)

    t = Timing()

    complete_trajectories = []
    num_to_process = 20

    while True:
        with t.timeit('loop'):
            with t.timeit('step'):
                actions = np.random.randint(0, agent.actor_critic.num_actions, params.num_envs)
                new_obs, rewards, dones, infos = multi_env.step(actions)

            with t.timeit('misc'):
                trajectory_buffer.add(observations, actions, dones)

                observations = main_observation(new_obs)

                num_steps_delta = num_env_steps(infos)
                num_steps += num_steps_delta

                complete_trajectories.extend(trajectory_buffer.complete_trajectories)
                trajectory_buffer.reset_trajectories()

            with t.timeit('train'):
                while len(complete_trajectories) > num_to_process:
                    training_steps = train_distance_net(
                        agent, complete_trajectories[:num_to_process], params, num_steps,
                    )
                    complete_trajectories = complete_trajectories[num_to_process:]

        loop_time.append(t.loop)
        advanced_steps.append(num_steps_delta)

        if num_steps % 100 == 0:
            avg_fps = sum(advanced_steps) / sum(loop_time)
            log.info('Step %d, avg. fps %.1f, training steps %d, timing: %s', num_steps, avg_fps, training_steps, t)


def train_distance(params, env_id):
    def make_env_func():
        e = create_env(env_id)
        e.seed(0)
        return e

    agent = AgentTMAX(make_env_func, params)
    agent.initialize()

    multi_env = None
    try:
        multi_env = MultiEnv(
            params.num_envs,
            params.num_workers,
            make_env_func=agent.make_env_func,
            stats_episodes=params.stats_episodes,
        )

        train_loop(agent, multi_env)
    except (Exception, KeyboardInterrupt, SystemExit):
        log.exception('Interrupt...')
    finally:
        log.info('Closing env...')
        if multi_env is not None:
            multi_env.close()

        agent.finalize()

    return 0


def main():
    args, params = parse_args_tmax(AgentTMAX.Params)
    status = train_distance(params, args.env)
    return status


if __name__ == '__main__':
    sys.exit(main())
