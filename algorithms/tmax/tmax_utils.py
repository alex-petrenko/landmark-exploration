from algorithms.arguments import parse_args

# values to use if not specified in the command line

DEFAULT_EXPERIMENT_NAME = 'tmax_v001'
DEFAULT_ENV = 'doom_maze'


def parse_args_tmax(params_cls):
    return parse_args(DEFAULT_ENV, DEFAULT_EXPERIMENT_NAME, params_cls)
