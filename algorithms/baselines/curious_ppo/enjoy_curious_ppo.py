import sys

import cv2

from algorithms.baselines.curious_ppo.curious_ppo_utils import parse_args_curious_ppo
from algorithms.baselines.curious_ppo.agent_curious_ppo import AgentCuriousPPO
from utils.envs.envs import create_env
from algorithms.exploit import run_policy_loop
from utils.utils import log


def enjoy(params, env_id, max_num_episodes=1000000, max_num_frames=1e9, fps=1500, show_automap=True):
    def make_env_func():
        e = create_env(env_id, mode='test', show_automap=show_automap)
        e.seed(0)
        return e

    agent = AgentCuriousPPO(make_env_func, params.load())
    env = make_env_func()

    # this helps with screen recording
    pause_at_the_beginning = False
    if pause_at_the_beginning:
        env.render()
        log.info('Press any key to start...')
        cv2.waitKey()

    return run_policy_loop(agent, env, max_num_episodes, fps, max_num_frames=max_num_frames, deterministic=False)


def main():
    args, params = parse_args_curious_ppo(AgentCuriousPPO.Params)
    return enjoy(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
