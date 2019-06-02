import sys

from utils.envs.envs import create_env


def main():
    env = create_env('doom_textured_random_goal_v2', mode='test', show_automap=True)
    return env.unwrapped.play_human_mode()


if __name__ == '__main__':
    sys.exit(main())
