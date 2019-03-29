from algorithms.arguments import parse_args

# values to use if not specified in the command line
from algorithms.curiosity.reachability_curiosity.reachability import ReachabilityBuffer
from algorithms.trajectory import Trajectory, TrajectoryBuffer

DEFAULT_EXPERIMENT_NAME = 'tmax_v020'
DEFAULT_ENV = 'doom_maze_very_sparse'


def parse_args_tmax(params_cls):
    return parse_args(DEFAULT_ENV, DEFAULT_EXPERIMENT_NAME, params_cls)


class TmaxMode:
    """
    EXPLORATION: looking for new landmarks/edges
    IDLE_EXPLORATION: explore + idle to train distance metric (for Montezuma, not needed for 3D mazes)
    LOCOMOTION: moving between landmarks in the graph
    """

    _num_modes = 3
    EXPLORATION, LOCOMOTION, IDLE_EXPLORATION = range(_num_modes)

    @staticmethod
    def all_modes():
        return list(range(TmaxMode._num_modes))

    @staticmethod
    def mode_name(mode):

        names = {
            TmaxMode.EXPLORATION: 'exploration',
            TmaxMode.LOCOMOTION: 'locomotion',
            TmaxMode.IDLE_EXPLORATION: 'idle_exploration',
        }
        return names[mode]


class TmaxReachabilityBuffer(ReachabilityBuffer):
    def skip(self, trajectory, i):
        """Override base class method."""
        # train reachability only on samples from exploration policy
        return trajectory.mode[i] != TmaxMode.EXPLORATION


class TmaxTrajectory(Trajectory):
    def __init__(self, env_idx):
        super().__init__(env_idx)
        self.mode = []

    def add(self, obs, action, **kwargs):
        super().add(obs, action, **kwargs)
        self.mode.append(kwargs['mode'])

    def add_frame(self, tr, i):
        self.add(tr.obs[i], tr.actions[i], mode=tr.mode[i])


class TmaxTrajectoryBuffer(TrajectoryBuffer):
    """Store trajectories for multiple parallel environments."""
    def __init__(self, num_envs):
        super().__init__(num_envs)
        self.current_trajectories = [TmaxTrajectory(env_idx) for env_idx in range(num_envs)]
        self.complete_trajectories = []

    def add(self, obs, actions, dones, **kwargs):
        assert len(obs) == len(actions)
        for env_idx in range(len(obs)):
            if dones[env_idx]:
                # finalize the trajectory and put it into a separate buffer
                self.complete_trajectories.append(self.current_trajectories[env_idx])
                self.current_trajectories[env_idx] = TmaxTrajectory(env_idx)
            else:
                self.current_trajectories[env_idx].add(obs[env_idx], actions[env_idx], mode=kwargs['modes'][env_idx])
