import numpy as np

from utils.utils import min_with_idx


def edge_weight(i1, i2, d):
    if d['loop_closure']:
        return 3
    if i2 < i1:
        return 3
    return 1


class Navigator:
    def __init__(self, agent):
        self.agent = agent
        self.params = agent.params
        self.distance_net = agent.distance

        self.current_landmarks = [0] * self.params.num_envs
        self.paths = [[] for _ in range(self.params.num_envs)]

        self.lost_localization_frames = [0] * self.params.num_envs

        # navigation parameters
        self.max_neighborhood = 4
        self.max_neighborhood_dist = 0.5
        self.max_lookahead = 7
        self.confidently_reachable = 0.1
        self.max_lost_localization = 10

    def reset(self, env_i, m):
        self.current_landmarks[env_i] = 0  # assuming we always start from the same location
        self.lost_localization_frames[env_i] = 0

        # reset shortest paths to the goal because we might have a new map or a new goal
        self.paths[env_i] = [None] * m.num_landmarks()

    def _ensure_paths_to_goal_calculated(self, maps, goals):
        for env_i in range(self.params.num_envs):
            m = maps[env_i]
            goal = goals[env_i]
            if m is None or goal is None:
                continue

            curr_landmark = self.current_landmarks[env_i]
            if self.paths[env_i][curr_landmark] is not None:
                # shortest path for this environment is already calculated
                continue

            path = m.get_path(curr_landmark, goal, edge_weight=edge_weight)
            assert path is not None and len(path) > 0

            curr_node = curr_landmark
            assert path[0] == curr_node
            for next_node in path[1:]:
                if self.paths[env_i][curr_node] is not None:
                    # next target for the rest of the path is already known
                    break

                self.paths[env_i][curr_node] = next_node
                curr_node = next_node

            assert path[-1] == goal
            self.paths[env_i][goal] = goal  # once we're already there let the path be trivial

    def _path_lookahead(self, env_i):
        curr_landmark = self.current_landmarks[env_i]
        lookahead = [curr_landmark]

        curr_node = curr_landmark
        for i in range(self.max_lookahead):
            next_node = self.paths[env_i][curr_node]
            lookahead.append(next_node)
            if curr_node == next_node:
                # reached the end of the path
                break

            curr_node = next_node

        return lookahead

    def _localize_path_lookahead(self, maps, obs, goals):
        self._ensure_paths_to_goal_calculated(maps, goals)

        # create a batch of all neighborhood observations from all envs for fast processing on GPU
        neighborhood_obs, neighborhood_hashes, current_obs = [], [], []
        neighbor_indices = []
        for env_i, m in enumerate(maps):
            if m is None or goals[env_i] is None:
                continue

            neighbors = self._path_lookahead(env_i)
            neighbor_indices.append(neighbors)
            neighborhood_obs.extend([m.get_observation(i) for i in neighbors])
            neighborhood_hashes.extend([m.get_hash(i) for i in neighbors])
            current_obs.extend([obs[env_i]] * len(neighbors))

        assert len(neighborhood_obs) == len(current_obs)
        assert len(neighborhood_obs) == len(neighborhood_hashes)

        distances = self.distance_net.distances_from_obs(
            self.agent.session,
            obs_first=neighborhood_obs, obs_second=current_obs,
            hashes_first=neighborhood_hashes, hashes_second=None,  # calculate curr obs hashes on the fly
        )

        lookahead_distances = [None] * self.params.num_envs
        j = 0
        for env_i, m in enumerate(maps):
            if m is None or goals[env_i] is None:
                continue

            j_next = j + len(neighbor_indices[env_i])
            distance = distances[j:j_next]
            j = j_next

            lookahead_distances[env_i] = distance

        return neighbor_indices, lookahead_distances

    def get_next_target(self, maps, obs, goals):
        """Returns indices of the next locomotion targets for all envs, or nones if we're lost."""
        neighbors, distances = self._localize_path_lookahead(maps, obs, goals)

        next_target = [None] * self.params.num_envs
        next_target_d = [None] * self.params.num_envs

        for env_i, m in enumerate(maps):
            if m is None or goals[env_i] is None:
                continue

            distance = distances[env_i]
            min_d, min_d_idx = min_with_idx(distance)
            closest_landmark = neighbors[env_i][min_d_idx]

            if min_d > self.max_neighborhood_dist:
                self.lost_localization_frames[env_i] += 1
                curr_landmark_on_the_path = 0
            else:
                self.lost_localization_frames[env_i] = 0
                # noinspection PyTypeChecker
                self.current_landmarks[env_i] = closest_landmark
                curr_landmark_on_the_path = min_d_idx

            if self.lost_localization_frames[env_i] > self.max_lost_localization:
                continue

            lookahead_path = neighbors[env_i][curr_landmark_on_the_path:]
            distance = distance[curr_landmark_on_the_path:]

            target_node = lookahead_path[0]
            target_d = distance[0]
            confidently_reachable = np.random.random() * 0.1 + 0.1

            if len(lookahead_path) > 1 and distance[1] < self.max_neighborhood_dist:
                target_node = lookahead_path[1]
                target_d = distance[1]
                for i, node in enumerate(lookahead_path[2:], start=2):
                    if distance[i] > confidently_reachable:
                        break
                    target_node = node
                    target_d = distance[i]

            # noinspection PyTypeChecker
            next_target[env_i] = target_node
            next_target_d[env_i] = target_d

        return next_target, next_target_d
