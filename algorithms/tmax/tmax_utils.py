from algorithms.arguments import parse_args

# values to use if not specified in the command line
from utils.envs.envs import create_env

DEFAULT_EXPERIMENT_NAME = 'tmax_v000'
DEFAULT_ENV = 'doom_basic'


def parse_args_tmax(params_cls):
    return parse_args(DEFAULT_ENV, DEFAULT_EXPERIMENT_NAME, params_cls)
