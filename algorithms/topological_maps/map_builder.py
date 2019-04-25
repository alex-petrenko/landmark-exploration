import math
import random
from functools import partial

import numpy as np

from algorithms.topological_maps.localization import Localizer
from algorithms.topological_maps.topological_map import hash_observation, TopologicalMap
from utils.timing import Timing
from utils.utils import log


class MapBuilder:
    def __init__(self, agent):
        self.agent = agent
        self.distance_net = self.agent.distance
        self.obs_encoder = self.distance_net.obs_encoder

        # map generation parameters
        self.max_duplicate_dist = 20
        self.duplicate_neighborhood = 5
        self.duplicate_threshold = 0.1

        self.shortcut_dist_threshold = 0.1
        self.shortcut_risk_threshold = 0.15
        self.min_shortcut_dist = 5
        self.shortcut_window = 10
        self.shortcuts_to_keep_fraction = 0.25  # fraction of the number of all nodes

    @staticmethod
    def _add_simple_path_to_map(m, traj, node_idx):
        obs, infos = traj.obs, traj.infos
        trajectory_idx = m.num_trajectories

        nodes = m.graph.nodes
        nodes[0]['info'] = infos[0]
        nodes[0]['embedding'] = None
        nodes[0]['traj_idx'] = 0
        nodes[0]['frame_idx'] = 0

        for i in range(1, len(traj)):
            idx = m.add_landmark(obs[i], infos[i], update_curr_landmark=True)
            assert node_idx[i] == -1
            assert idx >= i
            node_idx[i] = idx

            nodes[idx]['info'] = infos[i]
            nodes[idx]['embedding'] = None
            nodes[idx]['traj_idx'] = trajectory_idx
            nodes[idx]['frame_idx'] = i

        m.frame_to_node_idx[m.num_trajectories] = node_idx
        m.num_trajectories += 1

    def _calc_feature_vectors(self, m, traj, node_idx):
        obs = traj.obs

        obs_hashes = [hash_observation(o) for o in obs]
        self.obs_encoder.encode(self.agent.session, obs)

        assert len(traj) == len(obs_hashes)

        for i in range(len(traj)):
            obs_hash = obs_hashes[i]
            node = node_idx[i]
            m.graph.nodes[node]['embedding'] = self.obs_encoder.encoded_obs[obs_hash]

    def _calc_pairwise_distances(self, m):
        num_landmarks = m.num_landmarks()

        pairwise_distances = np.empty([num_landmarks, num_landmarks], np.float32)
        all_encoded_obs = [data['embedding'] for node, data in m.graph.nodes.data()]

        for i in range(num_landmarks):
            if i % 10 == 0:
                log.debug('Pairwise distances for %05d...', i)

            curr_obs = [all_encoded_obs[i]] * num_landmarks
            d = self.distance_net.distances(self.agent.session, curr_obs, all_encoded_obs)
            pairwise_distances[i, :] = d

        # induce symmetry
        for i in range(num_landmarks):
            for j in range(i + 1, num_landmarks):
                d = (pairwise_distances[i][j] + pairwise_distances[j][i]) * 0.5
                pairwise_distances[i][j] = pairwise_distances[j][i] = d

        return pairwise_distances

    def _sparsify_trajectory_map(self, m, traj, node_idx, pairwise_distances, keep_frames):
        to_delete = []
        duplicates = [[] for _ in range(m.num_landmarks())]

        for i in range(len(traj)):
            i_node = node_idx[i]
            if i_node in to_delete:
                continue

            for j in range(i + 1, min(len(traj), i + 1 + self.max_duplicate_dist)):
                j_node = node_idx[j]
                if j_node in to_delete or j in keep_frames:
                    continue

                d = pairwise_distances[i_node][j_node]
                if d > self.duplicate_threshold:
                    continue

                neighbor_dist = []
                for shift in range(-self.duplicate_neighborhood, self.duplicate_neighborhood + 1):
                    shifted_i, shifted_j = i + shift, j + shift
                    if shifted_i < 0 or shifted_i >= len(traj):
                        continue
                    if shifted_j < 0 or shifted_j >= len(traj):
                        continue

                    neighbor_dist.append(pairwise_distances[i_node][node_idx[shifted_j]])
                    neighbor_dist.append(pairwise_distances[node_idx[shifted_i]][j_node])

                if np.percentile(neighbor_dist, 50) < self.duplicate_threshold:
                    log.info('Duplicate landmark %d-%d (frames %d-%d)', i_node, j_node, i, j)
                    to_delete.append(j_node)
                    duplicates[i_node].append(j)
                    print(keep_frames)
                    assert j not in keep_frames
                else:
                    break

        new_map = TopologicalMap(
            m.get_observation(0), directed_graph=False, initial_info=m.graph.nodes[0]['info'],
        )
        new_map.num_trajectories = m.num_trajectories
        new_map.frame_to_node_idx = m.frame_to_node_idx
        new_map.new_episode()

        prev_traj_idx = 0
        prev_node = 0
        for node, data in m.graph.nodes.data():
            if node in to_delete:
                assert len(duplicates[node]) <= 0
                continue

            traj_idx = data.get('traj_idx', 0)
            assert traj_idx >= prev_traj_idx

            new_node = 0
            if node > 0:
                new_node = new_map.add_landmark(data['obs'], data['info'], update_curr_landmark=True)
                add_edge = traj_idx == prev_traj_idx
                if not add_edge:
                    new_map.remove_edges_from([(prev_node, new_node)])
                    log.info('Remove edge %r between different trajectories', [(prev_node, new_node)])

            for key, value in data.items():
                new_map.graph.nodes[new_node][key] = value

            # maintain correct frame to node correspondence
            if traj_idx == m.num_trajectories - 1:
                frame_idx = data['frame_idx']
                node_idx[frame_idx] = new_node

            for duplicate_idx in duplicates[node]:
                log.info('Node for frame %d is %d', duplicate_idx, new_node)
                node_idx[duplicate_idx] = new_node

            prev_traj_idx = traj_idx
            prev_node = node

        new_map.frame_to_node_idx[new_map.num_trajectories - 1] = node_idx

        log.debug('%d landmarks in the map after duplicate removal...', new_map.num_landmarks())
        return new_map

    def _shortcuts_distance(self, m, pairwise_distances, min_shortcut_dist, shortcut_window):
        shortcut_candidates = []

        nodes = m.graph.nodes
        for i, data in nodes.data():
            if i % 10 == 0:
                log.debug('Checking loop closures for %05d...', i)

            i_traj_idx = data.get('traj_idx', 0)

            for j in range(i, m.num_landmarks()):
                j_traj_idx = nodes[j].get('traj_idx', 0)

                if j - i < min_shortcut_dist and i_traj_idx == j_traj_idx:
                    # skip trivial shorcuts (close in time and from the same trajectory)
                    continue

                neighbors_dist = []

                d = pairwise_distances[i][j]
                if d > self.shortcut_dist_threshold:
                    continue

                # check how aligned the landmark neighborhoods are
                for shift in range(-shortcut_window, shortcut_window + 1):
                    shifted_i, shifted_j = i + shift, j + shift
                    if shifted_i < 0 or shifted_i >= m.num_landmarks():
                        continue
                    if shifted_j < 0 or shifted_j >= m.num_landmarks():
                        continue

                    shifted_i_traj_idx = nodes[shifted_i].get('traj_idx', 0)
                    shifted_j_traj_idx = nodes[shifted_j].get('traj_idx', 0)
                    if shifted_i_traj_idx != i_traj_idx or shifted_j_traj_idx != j_traj_idx:
                        continue

                    neighbors_dist.append(pairwise_distances[shifted_i][shifted_j])

                # the more aligned neighborhoods are, the less risk there is for the shortcut to be noise
                distance_percentile = np.percentile(neighbors_dist, 60)
                shorcut_risk = distance_percentile  # closer to 0 = better

                # calculate ground-truth distance (purely for diagnostic purposes)
                xi = nodes[i]['info']['pos']['agent_x']
                yi = nodes[i]['info']['pos']['agent_y']
                xj = nodes[j]['info']['pos']['agent_x']
                yj = nodes[j]['info']['pos']['agent_y']
                gt_dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

                shortcut = shorcut_risk, i, j, d, gt_dist
                assert i < j
                shortcut_candidates.append(shortcut)

        return shortcut_candidates

    def _add_shortcuts(self, m, pairwise_distances):
        shortcuts = self._shortcuts_distance(
            m, pairwise_distances, self.min_shortcut_dist, self.shortcut_window,
        )
        if len(shortcuts) <= 0:
            log.warning('Could not find any shortcuts')
            return

        random.shuffle(shortcuts)
        shortcut_risks = [s[0] for s in shortcuts]

        shortcuts_to_keep = int(self.shortcuts_to_keep_fraction * m.num_landmarks())

        keep = min(shortcuts_to_keep, len(shortcuts))
        percentile = (keep / len(shortcuts)) * 100
        max_risk = np.percentile(shortcut_risks, percentile)
        max_risk = min(max_risk, self.shortcut_risk_threshold)

        log.debug('Keep shortcuts with risk <= %.3f...', max_risk)
        shortcuts = [s for s in shortcuts if s[0] <= max_risk][:keep]
        shortcuts.sort(key=lambda x: x[-1], reverse=True)  # sort according to ground truth distance for logging

        log.debug('Kept %d shortcuts: %r...', len(shortcuts), shortcuts[:5])

        for shortcut in shortcuts:
            risk, i1, i2, d, coord_dist = shortcut
            m.add_edge(i1, i2, loop_closure=True)

    def add_trajectory_to_dense_map(self, existing_map, traj, keep_frames):
        t = Timing()

        m = existing_map
        m.new_episode()  # just in case

        node_idx = [-1] * len(traj)  # index map from trajectory frame to graph node idx
        node_idx[0] = 0  # first observation is always the same (we start from the same initial state)

        with t.timeit('create_initial_map'):
            self._add_simple_path_to_map(m, traj, node_idx)

        # precalculate feature vectors for the distances network
        with t.timeit('cache_feature_vectors'):
            self._calc_feature_vectors(m, traj, node_idx)

        with t.add_time('pairwise_distances'):
            pairwise_distances = self._calc_pairwise_distances(m)

        # delete very close (duplicate) landmarks from the map
        with t.timeit('sparsify'):
            m = self._sparsify_trajectory_map(m, traj, node_idx, pairwise_distances, keep_frames)

        with t.add_time('pairwise_distances'):
            pairwise_distances = self._calc_pairwise_distances(m)

        with t.timeit('loop_closures'):
            self._add_shortcuts(m, pairwise_distances)

        log.debug('Add trajectory to map, timing: %s', t)
        return m

    def add_trajectory_to_sparse_map(self, existing_map, traj):
        m = existing_map
        localizer = Localizer(self.agent.params)
        is_new_landmark = [False] * len(traj)  # is frame a novel landmark

        nodes = m.graph.nodes
        nodes[0]['traj_idx'] = 0
        nodes[0]['frame_idx'] = 0

        def new_landmark(_, new_landmark_idx, frame_idx):
            is_new_landmark[frame_idx] = True
            nodes[new_landmark_idx]['traj_idx'] = m.num_trajectories
            nodes[new_landmark_idx]['frame_idx'] = frame_idx

        for i in range(len(traj)):
            new_landmark_func = partial(new_landmark, frame_idx=i)
            obs = traj.obs[i]
            info = traj.infos[i]
            localizer.localize(
                self.agent.session, [obs], [info], [m], self.distance_net, on_new_landmark=new_landmark_func,
            )

        m.num_trajectories += 1
        return is_new_landmark
