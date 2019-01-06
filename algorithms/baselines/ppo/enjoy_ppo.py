import sys

import cv2

from algorithms.baselines.ppo.agent_ppo import AgentPPO
from algorithms.baselines.ppo.ppo_utils import parse_args_ppo
from algorithms.env_wrappers import create_env_args
from algorithms.exploit import run_policy_loop
from utils.utils import log


def enjoy(args, params, env_id, max_num_episodes=1000000, fps=30):
    def make_env_func():
        e = create_env_args(env_id, args, params)
        e.seed(0)
        return e

    agent = AgentPPO(make_env_func, params.load())
    env = make_env_func()

    # this helps with screen recording
    pause_at_the_beginning = False
    if pause_at_the_beginning:
        env.render()
        log.info('Press any key to start...')
        cv2.waitKey()

    return run_policy_loop(agent, env, max_num_episodes, fps, deterministic=False)


def main():
    args, params = parse_args_ppo(AgentPPO.Params)
    return enjoy(args, params, args.env_id)


if __name__ == '__main__':
    sys.exit(main())
