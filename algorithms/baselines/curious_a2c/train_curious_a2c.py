import sys

from algorithms.baselines.curious_a2c.agent_curious_a2c import AgentCuriousA2C
from algorithms.baselines.curious_a2c.curious_a2c_utils import parse_args_curious_a2c
from utils.envs.envs import create_env


def train(params, env_id):
    def make_env_func():
        return create_env(env_id, has_timer=True)

    agent = AgentCuriousA2C(make_env_func, params=params)
    agent.initialize()
    agent.learn()
    agent.finalize()
    return 0


def main():
    """Script entry point."""
    args, params = parse_args_curious_a2c(AgentCuriousA2C.Params)
    return train(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
