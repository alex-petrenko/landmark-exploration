class Trajectory:
    def __init__(self, env_idx):
        self.obs = []
        self.actions = []
        self.infos = []
        self.env_idx = env_idx

    def add(self, obs, action, info, **kwargs):
        self.obs.append(obs)
        self.actions.append(action)
        self.infos.append(info)

    def add_frame(self, tr, i):
        self.add(tr.obs[i], tr.actions[i], tr.infos[i])

    def add_trajectory(self, tr):
        self.obs.extend(tr.obs)
        self.actions.extend(tr.actions)
        self.infos.extend(tr.infos)

    def __len__(self):
        return len(self.obs)

    def obs_nbytes(self):
        if len(self) == 0:
            return 0
        obs_size = self.obs[0].nbytes
        return len(self) * obs_size


class TrajectoryBuffer:
    """Store trajectories for multiple parallel environments."""

    def __init__(self, num_envs):
        self.current_trajectories = [Trajectory(env_idx) for env_idx in range(num_envs)]
        self.complete_trajectories = []

    def reset_trajectories(self):
        """Discard old trajectories and start collecting new ones."""
        self.complete_trajectories = []

    def add(self, obs, actions, infos, dones):
        assert len(obs) == len(actions)
        for env_idx in range(len(obs)):
            self.current_trajectories[env_idx].add(obs[env_idx], actions[env_idx], infos[env_idx])

            if dones[env_idx]:
                # finalize the trajectory and put it into a separate buffer
                self.complete_trajectories.append(self.current_trajectories[env_idx])
                self.current_trajectories[env_idx] = Trajectory(env_idx)

    def obs_size(self):
        total_len = total_nbytes = 0
        for traj_buffer in [self.current_trajectories, self.complete_trajectories]:
            for traj in traj_buffer:
                total_len += len(traj)
                total_nbytes += traj.obs_nbytes()

        return total_len, total_nbytes

