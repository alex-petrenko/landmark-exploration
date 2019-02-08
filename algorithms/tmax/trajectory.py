class Trajectory:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.landmarks = [0]  # indices of observations marked as landmarks

    def add(self, obs, action):
        self.obs.append(obs)
        self.actions.append(action)

    def set_landmark(self):
        num_obs = len(self.obs)
        assert num_obs > 1
        assert len(self.landmarks) >= 1
        self.landmarks.append(num_obs - 1)


class TrajectoryBuffer:
    """Store trajectories for multiple parallel environments."""

    def __init__(self, num_envs):
        """For now we don't need anything except obs and actions."""
        self.current_trajectories = [Trajectory() for _ in range(num_envs)]
        self.complete_trajectories = []

    def reset_trajectories(self):
        """Discard old trajectories and start collecting new ones."""
        self.complete_trajectories = []

    def add(self, obs, actions, dones):
        assert len(obs) == len(actions)
        for env_idx in range(len(obs)):
            if dones[env_idx]:
                # finalize the trajectory and put it into a separate buffer
                self.complete_trajectories.append(self.current_trajectories[env_idx])
                self.current_trajectories[env_idx] = Trajectory()
            else:
                self.current_trajectories[env_idx].add(obs[env_idx], actions[env_idx])

    def set_landmark(self, env_idx):
        """Mark the current observation for env as a landmark. Can be used later for locomotion policy training."""
        self.current_trajectories[env_idx].set_landmark()
