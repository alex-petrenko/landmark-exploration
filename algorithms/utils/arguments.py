import argparse
import sys

from utils.utils import log, get_experiment_name


def parse_args(default_env, default_experiment_name, params_cls, argv=None):
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # common args
    parser.add_argument('--experiment', type=str, default=None)
    parser.add_argument('--env', type=str, default=default_env)

    # params object args
    params_cls.add_cli_args(parser)

    args = parser.parse_args(argv)

    experiment = args.experiment
    if experiment is None:
        experiment = get_experiment_name(args.env, default_experiment_name)

    params = params_cls(experiment)
    params.set_command_line(sys.argv)
    params.update(args)

    log.info('Config:')
    for arg in vars(args):
        log.info('%s %r', arg, getattr(args, arg))

    return args, params

