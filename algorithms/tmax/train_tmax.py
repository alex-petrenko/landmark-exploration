import sys
from functools import partial

from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
from utils.envs.envs import create_env


def train(params, env_id):
    agent = AgentTMAX(partial(create_env, env=env_id), params=params)
    agent.initialize()
    status = agent.learn()
    agent.finalize()
    return status


def main():
    """Script entry point."""
    args, params = parse_args_tmax(AgentTMAX.Params)
    return train(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
