import random

import numpy as np

from algorithms.utils.algo_utils import EPS
from utils.utils import min_with_idx, log, scale_to_range


def default_edge_weight(i1, i2, d):
    if d['loop_closure']:
        return 300
    if i2 < i1:
        return 300
    return 1


class Navigator:
    def __init__(self, agent):
        self.agent = agent
        self.params = agent.params
        self.distance_net = agent.distance

        self.current_landmarks = [0] * self.params.num_envs
        self.last_made_progress = [0] * self.params.num_envs

        self.paths = [[] for _ in range(self.params.num_envs)]

        self.lost_localization_frames = [0] * self.params.num_envs

        # navigation parameters
        self.max_neighborhood_dist = 0.5
        self.max_lookahead = 7
        self.confidently_reachable = 0.05
        self.max_lost_localization = 40
        self.max_no_progress = 50

        self.edge_weight = default_edge_weight

    def reset(self, env_i, m):
        self.current_landmarks[env_i] = 0  # assuming we always start from the same location
        self.last_made_progress[env_i] = 0
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

            path = m.get_path(curr_landmark, goal, edge_weight=self.edge_weight)

            if path is None or len(path) <= 0:
                log.error('Nodes: %r', list(m.graph.nodes))
                log.error('Path %r', path)
                log.error('Current landmark: %d', curr_landmark)
                log.error('Goal: %d', goal)

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
        neighbor_indices = [[]] * len(maps)
        neighbor_diff = []
        for env_i, m in enumerate(maps):
            if m is None or goals[env_i] is None:
                continue

            neighbors = self._path_lookahead(env_i)
            neighbor_indices[env_i] = neighbors

            # not robust to very large diffs in node numbers (e.g. with loop closures)
            # curr_landmark = self.current_landmarks[env_i]
            # neighbor_diff.extend([n - curr_landmark for n in neighbors])
            neighbor_diff.extend(np.arange(len(neighbors)))
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

        c_frames = 0  # set to 0 to disable
        if c_frames != 0:  # mix of both num_frames_diff and distances
            trajectory_penalty = 0.1
            new_dist = 1 * np.array(distances) + \
                c_frames * np.array(neighbor_diff) * trajectory_penalty / self.max_lookahead

            # keep this new_dist param in the same (min,max) range as distances above.
            # Downstream thresholds are dependent on it
            distances = scale_to_range(new_dist, np.min(distances), np.max(distances))
            # distances = new_dist
            # print('new metric {0}'.format(distances))

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

    def get_next_target(self, maps, obs, goals, episode_frames):
        """Returns indices of the next locomotion targets for all envs, or nones if we're lost."""
        neighbors, distances = self._localize_path_lookahead(maps, obs, goals)

        next_target = [None] * self.params.num_envs
        next_target_d = [None] * self.params.num_envs

        for env_i, m in enumerate(maps):
            if m is None or goals[env_i] is None:
                continue

            # log.info('Neighbors: %r', neighbors[env_i])
            # log.info('Distances: %s', ', '.join([f'{d:.3f}' for d in distances[env_i]]))

            distance = distances[env_i]
            min_d, min_d_idx = min_with_idx(distance[:2])
            closest_landmark = neighbors[env_i][min_d_idx]
            prev_landmark = self.current_landmarks[env_i]

            if min_d > self.max_neighborhood_dist:
                self.lost_localization_frames[env_i] += 1
                curr_landmark_on_the_path = 0
            else:
                self.lost_localization_frames[env_i] = 0

                if min_d_idx == 0 and len(distance) > 1 and distance[1] < random.random() * 0.04:
                    # current landmark (distance[0]) is the closest, but next landmark is also super close
                    # set current landmark to be the next landmark on the path to make some progress
                    min_d_idx = 1
                    closest_landmark = neighbors[env_i][min_d_idx]

                self.current_landmarks[env_i] = closest_landmark
                curr_landmark_on_the_path = min_d_idx

            if self.lost_localization_frames[env_i] > self.max_lost_localization:
                continue

            lookahead_path = neighbors[env_i][curr_landmark_on_the_path:]
            distance = distance[curr_landmark_on_the_path:]

            target_node = lookahead_path[0]
            target_d = distance[0]
            # confidently_reachable = np.random.random() * 0.05 + 0.01
            # confidently_reachable = self.confidently_reachable

            if len(maps) <= 1:
                # debug
                log.info('Curr landmark %d, path %r', self.current_landmarks[env_i], lookahead_path)
                log.info('Distances %r', [f'{d:.3f}' for d in distance])

            if len(lookahead_path) > 1 and distance[1] < self.max_neighborhood_dist:
                if distance[1] < 2 * self.confidently_reachable or random.random() < 0.5:
                    target_node = lookahead_path[1]
                    target_d = distance[1]

                # for i, node in enumerate(lookahead_path[2:3], start=2):
                #     if distance[i] > confidently_reachable:
                #         break
                #     target_node = node
                #     target_d = distance[i]
            # elif len(lookahead_path) > 1 and distance[1] < self.max_neighborhood_dist:
            #     if random.random() < 0.3:
            #         target_node = lookahead_path[1]
            #         target_d = distance[1]

            # noinspection PyTypeChecker
            next_target[env_i] = target_node
            next_target_d[env_i] = target_d

            if prev_landmark != self.current_landmarks[env_i]:
                self.last_made_progress[env_i] = episode_frames[env_i]

            since_last_progress = episode_frames[env_i] - self.last_made_progress[env_i]
            if since_last_progress > self.max_no_progress:
                log.warning(
                    'Agent %d did not make any progress in %d frames, locomotion failed',
                    env_i, since_last_progress,
                )
                next_target[env_i] = None

        return next_target, next_target_d


class NavigatorNaive(Navigator):
    """Just replaying the actions."""

    def __init__(self, agent):
        super().__init__(agent)
        self.next_action_to_take = [0] * self.params.num_envs
        self.next_target = [0] * self.params.num_envs

        def edge_weight(i1, i2, d):
            """Action replay can only use forward edges and no loop closures."""
            if d['loop_closure']:
                return 1e9
            if i2 < i1:
                return 1e9
            return 1

        self.edge_weight = edge_weight

    def reset(self, env_i, m):
        super().reset(env_i, m)
        self.next_action_to_take = [0] * self.params.num_envs
        self.next_target = [0] * self.params.num_envs

    def get_next_target(self, maps, obs, goals, episode_frames):
        self._ensure_paths_to_goal_calculated(maps, goals)

        next_target = [None] * self.params.num_envs
        next_target_d = [None] * self.params.num_envs

        for env_i, m in enumerate(maps):
            if m is None or goals[env_i] is None:
                continue

            lookahead = self._path_lookahead(env_i)
            assert lookahead[0] == self.current_landmarks[env_i]

            action = 0
            self.next_target[env_i] = self.current_landmarks[env_i]

            if len(lookahead) > 1:
                self.next_target[env_i] = lookahead[1]
                action = m.graph.adj[lookahead[0]][lookahead[1]]['action']

            self.next_action_to_take[env_i] = action

            next_target[env_i] = self.current_landmarks[env_i]
            next_target_d[env_i] = EPS

        return next_target, next_target_d

    def replay_action(self, env_indices):
        actions = np.zeros(len(env_indices), np.int32)

        for env_i in env_indices:
            actions[env_i] = self.next_action_to_take[env_i]
            self.current_landmarks[env_i] = self.next_target[env_i]

        return actions
