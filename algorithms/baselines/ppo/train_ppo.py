import sys

from algorithms.baselines.ppo.agent_ppo import AgentPPO
from algorithms.baselines.ppo.ppo_utils import parse_args_ppo
from utils.envs.envs import create_env


def train(ppo_params, env_id):
    def make_env_func():
        return create_env(env_id)

    agent = AgentPPO(make_env_func, params=ppo_params)
    agent.initialize()
    agent.learn()
    agent.finalize()
    return 0


def main():
    """Script entry point."""
    args, params = parse_args_ppo(AgentPPO.Params)
    return train(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
