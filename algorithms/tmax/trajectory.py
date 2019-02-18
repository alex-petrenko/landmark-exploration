class Trajectory:
    def __init__(self, env_idx):
        self.obs = []
        self.actions = []
        self.current_landmark_idx = []  # indices of closest landmarks
        self.neighbor_indices = []  # indices of neighbors in the graph
        self.landmarks = []  # indices of observations marked as landmarks
        self.env_idx = env_idx

    def add(self, obs, action, current_landmark_idx, neighbor_indices, is_landmark):
        self.obs.append(obs)
        self.actions.append(action)
        self.current_landmark_idx.append(current_landmark_idx)
        self.neighbor_indices.append(neighbor_indices)
        if is_landmark:
            self._set_landmark()

    def _set_landmark(self):
        num_obs = len(self.obs)
        self.landmarks.append(num_obs - 1)

    def __len__(self):
        return len(self.obs)


class TrajectoryBuffer:
    """Store trajectories for multiple parallel environments."""

    def __init__(self, num_envs):
        """For now we don't need anything except obs and actions."""
        self.current_trajectories = [Trajectory(env_idx) for env_idx in range(num_envs)]
        self.complete_trajectories = []

    def reset_trajectories(self):
        """Discard old trajectories and start collecting new ones."""
        self.complete_trajectories = []

    def add(self, obs, actions, dones, maps, is_landmark):
        assert len(obs) == len(actions)
        for env_idx in range(len(obs)):
            if dones[env_idx]:
                # finalize the trajectory and put it into a separate buffer
                self.complete_trajectories.append(self.current_trajectories[env_idx])
                self.current_trajectories[env_idx] = Trajectory(env_idx)
            else:
                self.current_trajectories[env_idx].add(
                    obs[env_idx],
                    actions[env_idx],
                    maps[env_idx].curr_landmark_idx,
                    maps[env_idx].neighbor_indices(),
                    is_landmark[env_idx],
                )
