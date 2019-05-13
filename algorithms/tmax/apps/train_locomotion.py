import sys
from collections import deque

import numpy as np
import tensorflow as tf

from algorithms.multi_env import MultiEnv
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.locomotion import LocomotionBuffer
from algorithms.tmax.tmax_utils import parse_args_tmax, TmaxTrajectoryBuffer
from algorithms.utils.algo_utils import main_observation, num_env_steps
from utils.envs.envs import create_env
from utils.timing import Timing
from utils.utils import log


def calc_test_error(agent, data, params, env_steps, bn_training=False):
    log.info('Calculating test error...')

    t = Timing()
    batch_size = params.locomotion_experience_replay_batch
    locomotion = agent.locomotion
    loco_step = locomotion.step.eval(session=agent.session)

    with t.timeit('test_error'):
        losses, reg_losses, correct = [], [], []

        obs_prev, obs_curr, obs_goal = data.buffer.obs_prev, data.buffer.obs_curr, data.buffer.obs_goal
        actions = data.buffer.actions

        for i in range(0, len(obs_curr) - 1, batch_size):
            start, end = i, i + batch_size

            loss, reg_loss, is_correct = agent.session.run(
                [locomotion.loss, locomotion.reg_loss, locomotion.correct],
                feed_dict={
                    locomotion.ph_obs_prev: obs_prev[start:end],
                    locomotion.ph_obs_curr: obs_curr[start:end],
                    locomotion.ph_obs_goal: obs_goal[start:end],
                    locomotion.ph_actions: actions[start:end],
                    locomotion.ph_is_training: bn_training,
                }
            )

            losses.append(loss)
            reg_losses.append(reg_loss)
            correct.append(is_correct)

        avg_loss = np.mean(losses)
        avg_reg_loss = np.mean(reg_losses)
        avg_correct = np.mean(correct)

        log.info(
            'Avg loss at %d steps is %.3f (reg %.3f, correct %.3f)', loco_step, avg_loss, avg_reg_loss, avg_correct,
        )

        if not bn_training:
            summary_obj_env_steps = tf.Summary()
            summary_obj_env_steps.value.add(tag='locomotion/test_loss_env_steps', simple_value=avg_loss)
            summary_obj_env_steps.value.add(tag='locomotion/test_correct_env_steps', simple_value=avg_correct)
            agent.summary_writer.add_summary(summary_obj_env_steps, env_steps)

            summary_obj_training_steps = tf.Summary()
            summary_obj_training_steps.value.add(tag='locomotion/test_loss_train_steps', simple_value=avg_loss)
            summary_obj_training_steps.value.add(tag='locomotion/test_correct_train_steps', simple_value=avg_correct)
            agent.summary_writer.add_summary(summary_obj_training_steps, loco_step)

            agent.summary_writer.flush()

    log.debug('Took %s', t)


def train_locomotion_net(agent, data, params, env_steps):
    num_epochs = params.locomotion_experience_replay_epochs

    summary = None
    prev_loss = 1e10
    batch_size = params.locomotion_experience_replay_batch
    locomotion = agent.locomotion
    loco_step = locomotion.step.eval(session=agent.session)

    log.info('Training loco_her %d pairs, batch %d, epochs %d', len(data.buffer), batch_size, num_epochs)
    t = Timing()

    for epoch in range(num_epochs):
        log.info('Epoch %d...', epoch + 1)

        with t.timeit('shuffle'):
            data.shuffle_data()
        log.info('Shuffling locomotion data took %s', t)

        losses = []

        obs_prev, obs_curr, obs_goal = data.buffer.obs_prev, data.buffer.obs_curr, data.buffer.obs_goal
        actions = data.buffer.actions

        for i in range(0, len(obs_curr) - 1, batch_size):
            # noinspection PyProtectedMember
            with_summaries = agent._should_write_summaries(loco_step) and summary is None
            summaries = [agent.loco_summaries] if with_summaries else []

            start, end = i, i + batch_size

            objectives = [locomotion.loss, locomotion.train_loco]

            result = agent.session.run(
                objectives + summaries,
                feed_dict={
                    locomotion.ph_obs_prev: obs_prev[start:end],
                    locomotion.ph_obs_curr: obs_curr[start:end],
                    locomotion.ph_obs_goal: obs_goal[start:end],
                    locomotion.ph_actions: actions[start:end],
                    locomotion.ph_is_training: True,
                }
            )

            loco_step += 1
            # noinspection PyProtectedMember
            agent._maybe_save(loco_step, env_steps)

            losses.append(result[0])

            if with_summaries:
                summary = result[-1]
                agent.summary_writer.add_summary(summary, global_step=env_steps)

        # check loss improvement at the end of each epoch, early stop if necessary
        avg_loss = np.mean(losses)
        log.info('Avg train loss is %.3f', avg_loss)

        if avg_loss >= prev_loss:
            log.info('Stopping loco_her after %d epochs because locomotion did not improve', epoch)
            log.info('Was %.4f now %.4f, ratio %.3f', prev_loss, avg_loss, avg_loss / prev_loss)
            break

        prev_loss = avg_loss

    return loco_step


def train_loop(agent, multi_env):
    params = agent.params

    observations = main_observation(multi_env.reset())
    infos = multi_env.info()

    trajectory_buffer = TmaxTrajectoryBuffer(multi_env.num_envs)
    locomotion_buffer = LocomotionBuffer(params)

    num_test_data = 5000
    locomotion_buffer_test = LocomotionBuffer(params)

    step, env_steps = agent.session.run([agent.locomotion.step, agent.total_env_steps])

    loop_time = deque([], maxlen=2500)
    advanced_steps = deque([], maxlen=2500)

    t = Timing()

    while True:
        with t.timeit('loop'):
            with t.timeit('step'):
                actions = np.random.randint(0, agent.actor_critic.num_actions, params.num_envs)
                new_obs, rewards, dones, new_infos = multi_env.step(actions)

            with t.timeit('misc'):
                trajectory_buffer.add(
                    observations, actions, infos, dones, tmax_mgr=agent.tmax_mgr, is_random=[True] * params.num_envs,
                )

                observations = main_observation(new_obs)
                infos = new_infos

                num_steps_delta = num_env_steps(infos)
                env_steps += num_steps_delta

            with t.timeit('train'):
                locomotion_buffer.extract_data(trajectory_buffer.complete_trajectories)
                trajectory_buffer.reset_trajectories()

                if len(locomotion_buffer.buffer) >= params.locomotion_experience_replay_buffer:
                    if len(locomotion_buffer_test.buffer) <= 0:
                        log.info('Prepare test data that we will never see during training...')
                        locomotion_buffer.shuffle_data()
                        locomotion_buffer_test.buffer.add_buff(locomotion_buffer.buffer, max_to_add=num_test_data)

                        # noinspection PyProtectedMember
                        log.info(
                            'Test buffer size %d, capacity %d',
                            locomotion_buffer_test.buffer._size, locomotion_buffer_test.buffer._capacity,
                        )
                    else:
                        step = train_locomotion_net(agent, locomotion_buffer, params, env_steps)

                    locomotion_buffer.reset()
                    calc_test_error(agent, locomotion_buffer_test, params, env_steps)
                    calc_test_error(agent, locomotion_buffer_test, params, env_steps, bn_training=True)

            if t.train > 1.0:
                log.debug('Train time: %s', t)

        loop_time.append(t.loop)
        advanced_steps.append(num_steps_delta)

        if env_steps % 100 == 0:
            avg_fps = sum(advanced_steps) / sum(loop_time)
            log.info('Step %d, avg. fps %.1f, training steps %d, timing: %s', env_steps, avg_fps, step, t)


def train_locomotion(params, env_id):
    def make_env_func():
        e = create_env(env_id, episode_horizon=params.episode_horizon)
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
    status = train_locomotion(params, args.env)
    return status


if __name__ == '__main__':
    sys.exit(main())
