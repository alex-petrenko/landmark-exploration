from algorithms.arguments import parse_args

# values to use if not specified in the command line

DEFAULT_EXPERIMENT_NAME = 'tmax_v017'
DEFAULT_ENV = 'doom_maze_very_sparse'


def parse_args_tmax(params_cls):
    return parse_args(DEFAULT_ENV, DEFAULT_EXPERIMENT_NAME, params_cls)


class TmaxMode:
    """
    EXPLORATION: looking for new landmarks/edges
    IDLE_EXPLORATION: explore + idle to train distance metric (for Montezuma, not needed for 3D mazes)
    LOCOMOTION: moving between landmarks in the graph

    Replace this with enum?
    """

    EXPLORATION, LOCOMOTION, IDLE_EXPLORATION = range(3)

    @staticmethod
    def mode_name(mode):

        names = {
            TmaxMode.EXPLORATION: 'exploration',
            TmaxMode.LOCOMOTION: 'locomotion',
            TmaxMode.IDLE_EXPLORATION: 'idle_exploration',
        }
        return names[mode]
