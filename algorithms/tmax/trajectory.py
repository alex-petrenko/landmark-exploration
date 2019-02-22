class Trajectory:
    def __init__(self, env_idx):
        self.obs = []
        self.actions = []
        self.modes = []
        self.target_idx = []
        self.curr_landmark_idx = []
        self.is_landmark = []
        self.deliberate_action = []
        self.env_idx = env_idx

    def add(self, obs, action, mode, target_idx, curr_landmark_idx, is_landmark, deliberate_action):
        self.obs.append(obs)
        self.actions.append(action)
        self.modes.append(mode)
        self.target_idx.append(target_idx)
        self.curr_landmark_idx.append(curr_landmark_idx)
        self.is_landmark.append(is_landmark)
        self.deliberate_action.append(deliberate_action)

    def add_frame(self, tr, i):
        self.add(
            tr.obs[i], tr.actions[i],
            tr.modes[i],
            tr.target_idx[i], tr.curr_landmark_idx[i], tr.is_landmark[i],
            tr.deliberate_action[i],
        )

    def __len__(self):
        return len(self.obs)


class TrajectoryBuffer:
    """Store trajectories for multiple parallel environments."""

    def __init__(self, num_envs):
        self.current_trajectories = [Trajectory(env_idx) for env_idx in range(num_envs)]
        self.complete_trajectories = []

    def reset_trajectories(self):
        """Discard old trajectories and start collecting new ones."""
        self.complete_trajectories = []

    def add(self, obs, actions, dones, tmax_mgr):
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
                    tmax_mgr.mode[env_idx],
                    tmax_mgr.locomotion_targets[env_idx],
                    tmax_mgr.maps[env_idx].curr_landmark_idx,
                    tmax_mgr.is_landmark[env_idx],
                    tmax_mgr.deliberate_action[env_idx],
                )
