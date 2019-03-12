import sys

from algorithms.multi_env import MultiEnv
from utils.envs.dmlab.dmlab_gym import DmlabGymEnv
from utils.utils import log


def main():
    def make_env():
        env = DmlabGymEnv()
        return env

    num_envs = 64
    num_workers = 16
    multi_env = MultiEnv(num_envs, num_workers, make_env, stats_episodes=100)
    num_resets = 0

    try:
        while True:
            multi_env.reset()
            num_resets += 1
            num_envs_generated = num_resets * num_envs
            log.info('Generated %d environments...', num_envs_generated)
    except (Exception, KeyboardInterrupt, SystemExit):
        log.exception('Interrupt...')
    finally:
        log.info('Closing env...')
        multi_env.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
