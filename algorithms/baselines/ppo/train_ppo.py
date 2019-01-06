import sys

from algorithms.baselines.ppo.agent_ppo import AgentPPO
from algorithms.baselines.ppo.ppo_utils import parse_args_ppo
from algorithms.env_wrappers import create_env_args


def train(args, params, env_id):
    def make_env_func():
        return create_env_args(env_id, args, params)

    agent = AgentPPO(make_env_func, params=params)
    agent.initialize()
    agent.learn()
    agent.finalize()
    return 0


def main():
    args, params = parse_args_ppo(AgentPPO.Params)
    return train(args, params, args.env_id)


if __name__ == '__main__':
    sys.exit(main())
