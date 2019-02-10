import argparse
import sys

from algorithms.agent import AgentRandom
from algorithms.baselines.a2c.agent_a2c import AgentA2C
from algorithms.baselines.ppo.agent_ppo import AgentPPO
from algorithms.curious_a2c.agent_curious_a2c import AgentCuriousA2C
from utils.utils import log, get_experiment_name


def parse_args(default_env, default_experiment_name, params_cls):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # common args
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--env', type=str, default=default_env)
    parser.add_argument('--model', type=str, default=None)

    # params object args
    params_cls.add_cli_args(parser)

    args = parser.parse_args()

    experiment = args.experiment
    if experiment is None:
        experiment = get_experiment_name(args.env, default_experiment_name)

    params = params_cls(experiment, args.env)
    params.set_command_line(sys.argv)
    params.update(args)

    log.info('Config:')
    for arg in vars(args):
        log.info('%s %r', arg, getattr(args, arg))

    return args, params


def parse_model():
    """Returns the agent class defined by the argument --model"""
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # common args
    parser.add_argument('--model', type=str, default=None)

    args, _ = parser.parse_known_args()

    if args.model == 'ppo':
        agent_cls = AgentPPO
    elif args.model == 'a2c':
        agent_cls = AgentA2C
    elif args.model == 'curious_a2c':
        agent_cls = AgentCuriousA2C
    elif args.model == 'random':
        agent_cls = AgentRandom
    else:
        raise Exception('Unsupported model {0}'.format(args.model))

    return agent_cls
