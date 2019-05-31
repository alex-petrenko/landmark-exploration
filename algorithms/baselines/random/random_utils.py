from algorithms.utils.arguments import parse_args

# values to use if not specified in the command line
DEFAULT_EXPERIMENT_NAME = 'random_v000'
DEFAULT_ENV = 'doom_basic'


def parse_args_random(params_cls):
    return parse_args(DEFAULT_ENV, DEFAULT_EXPERIMENT_NAME, params_cls)
