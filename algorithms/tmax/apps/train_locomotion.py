import sys
from collections import deque

import numpy as np

from algorithms.algo_utils import main_observation, num_env_steps
from algorithms.multi_env import MultiEnv
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.locomotion import LocomotionBuffer
from algorithms.tmax.tmax_utils import parse_args_tmax, TmaxTrajectoryBuffer
from utils.envs.envs import create_env
from utils.timing import Timing
from utils.utils import log


# noinspection PyProtectedMember
def train_loop(agent, multi_env):
    params = agent.params

    observations = main_observation(multi_env.reset())

    trajectory_buffer = TmaxTrajectoryBuffer(multi_env.num_envs)
    locomotion_buffer = LocomotionBuffer(params)

    num_steps = 0
    training_steps = 0
    fps = deque([], maxlen=1000)
    t = Timing()

    while True:
        with t.timeit('loop'):
            actions = np.random.randint(0, agent.actor_critic.num_actions, params.num_envs)
            new_obs, rewards, dones, infos = multi_env.step(actions)

            trajectory_buffer.add(observations, actions, dones, tmax_mgr=agent.tmax_mgr)

            observations = main_observation(new_obs)

            num_steps_delta = num_env_steps(infos)
            num_steps += num_steps_delta

            locomotion_buffer.extract_data(trajectory_buffer.complete_trajectories)

            if len(locomotion_buffer.buffer) >= params.locomotion_experience_replay_buffer:
                training_steps = agent._maybe_train_locomotion_experience_replay(locomotion_buffer, num_steps)
                locomotion_buffer.reset()

            trajectory_buffer.reset_trajectories()

        if num_steps % 100 == 0:
            fps.append(num_steps_delta / t.loop)
            avg_fps = np.mean(fps)
            log.info('Step %d, avg. fps %.1f, training steps %d', num_steps, avg_fps, training_steps)


def train_locomotion(params, env_id):
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
    status = train_locomotion(params, args.env)
    return status


if __name__ == '__main__':
    sys.exit(main())
