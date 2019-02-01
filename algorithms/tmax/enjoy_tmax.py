import sys

import cv2

from algorithms.exploit import run_policy_loop
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
from utils.envs.envs import create_env
from utils.utils import log


def enjoy(params, env_id, max_num_episodes=1000000, max_num_frames=None, fps=1500):
    def make_env_func():
        e = create_env(env_id, mode='test')
        e.seed(0)
        return e

    agent = AgentTMAX(make_env_func, params.load())
    env = make_env_func()

    # this helps with screen recording
    pause_at_the_beginning = False
    if pause_at_the_beginning:
        env.render()
        log.info('Press any key to start...')
        cv2.waitKey()

    return run_policy_loop(agent, env, max_num_episodes, fps, max_num_frames=max_num_frames, deterministic=False)


def main():
    args, params = parse_args_tmax(AgentTMAX.Params)
    return enjoy(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
