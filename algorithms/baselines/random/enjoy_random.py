import sys

import cv2

from algorithms.baselines.random.agent_random import AgentRandom
from algorithms.baselines.random.random_utils import parse_args_random  # TODO:fill
from utils.envs.envs import create_env
from algorithms.utils.exploit import run_policy_loop
from utils.utils import log


def enjoy(params, env_id, max_num_episodes=1000000, max_num_frames=1e9, fps=20):
    def make_env_func():
        e = create_env(env_id, mode='test', skip_frames=True)
        e.seed(0)
        return e

    agent = AgentRandom(make_env_func, params.load())
    env = make_env_func()

    # this helps with screen recording
    pause_at_the_beginning = False
    if pause_at_the_beginning:
        env.render()
        log.info('Press any key to start...')
        cv2.waitKey()

    return run_policy_loop(agent, env, max_num_episodes, fps, max_num_frames=max_num_frames, deterministic=False)


def main():
    args, params = parse_args_random(AgentRandom.Params) # TODO: see random.params class
    return enjoy(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
