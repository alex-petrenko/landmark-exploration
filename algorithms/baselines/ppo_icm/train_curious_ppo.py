import sys

from algorithms.baselines.ppo_icm.curious_ppo_utils import parse_args_curious_ppo
from algorithms.baselines.ppo_icm.agent_curious_ppo import AgentCuriousPPO
from utils.envs.envs import create_env


def train(curious_ppo_params, env_id):
    def make_env_func():
        return create_env(env_id)

    agent = AgentCuriousPPO(make_env_func, params=curious_ppo_params)
    agent.initialize()
    status = agent.learn()
    agent.finalize()
    return status


def main():
    """Script entry point."""
    args, params = parse_args_curious_ppo(AgentCuriousPPO.Params)
    return train(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
