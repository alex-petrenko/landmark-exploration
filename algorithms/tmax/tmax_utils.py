from algorithms.arguments import parse_args

# values to use if not specified in the command line

DEFAULT_EXPERIMENT_NAME = 'tmax_v012'
DEFAULT_ENV = 'doom_maze_very_sparse'


def parse_args_tmax(params_cls):
    return parse_args(DEFAULT_ENV, DEFAULT_EXPERIMENT_NAME, params_cls)


class TmaxMode:
    """
    EXPLORATION: explore + idle to train distance metric (for Montezuma, not needed for 3D mazes)
    LOCOMOTION: moving between landmarks in the graph
    SEARCH: looking for new landmarks/edges

    We probably need better names for these.
    """
    EXPLORATION, LOCOMOTION, SEARCH = range(3)
