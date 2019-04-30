import math
import random
from functools import partial

import numpy as np

from algorithms.tmax.navigator import edge_weight
from algorithms.topological_maps.localization import Localizer
from algorithms.topological_maps.topological_map import hash_observation
from utils.timing import Timing
from utils.utils import log, max_with_idx


class MapBuilder:
    def __init__(self, agent):
        self.agent = agent
        self.distance_net = self.agent.distance
        self.obs_encoder = self.distance_net.obs_encoder

        # map generation parameters
        self.max_duplicate_dist = 1
        self.duplicate_neighborhood = 5
        self.duplicate_threshold = 0.05

        self.shortcut_dist_threshold = 0.1
        self.shortcut_risk_threshold = 0.1
        self.min_shortcut_dist = 5
        self.shortcut_window = 10
        self.shortcuts_to_keep_fraction = 0.25  # fraction of the number of all nodes

    def _calc_pairwise_distances(self, obs_embeddings):
        num_embeddings = len(obs_embeddings)

        pairwise_distances = np.empty([num_embeddings, num_embeddings], np.float32)

        for i in range(num_embeddings):
            if i % 10 == 0:
                log.debug('Pairwise distances for %05d...', i)

            curr_obs = [obs_embeddings[i]] * num_embeddings
            d = self.distance_net.distances(self.agent.session, curr_obs, obs_embeddings)
            pairwise_distances[i, :] = d

        # induce symmetry
        for i in range(num_embeddings):
            for j in range(i + 1, num_embeddings):
                d = (pairwise_distances[i][j] + pairwise_distances[j][i]) * 0.5
                pairwise_distances[i][j] = pairwise_distances[j][i] = d

        return pairwise_distances

    def _calc_embeddings(self, observations):
        obs_hashes = [hash_observation(o) for o in observations]
        self.obs_encoder.encode(self.agent.session, observations)

        assert len(observations) == len(obs_hashes)

        embeddings = [None] * len(observations)
        for i in range(len(observations)):
            obs_hash = obs_hashes[i]
            embeddings[i] = self.obs_encoder.encoded_obs[obs_hash]

        return embeddings

    def sparsify_trajectory(self, traj):
        obs = traj.obs
        obs_embeddings = self._calc_embeddings(obs)
        pairwise_distances = self._calc_pairwise_distances(obs_embeddings)

        to_delete = set()

        for i in range(len(traj)):
            if i in to_delete:
                continue

            for j in range(i + 1, len(traj)):
                if j in to_delete:
                    continue

                if np.array_equal(traj.obs[i], traj.obs[j]):
                    log.info('Frames %d and %d are exactly the same!', i, j)
                    to_delete.add(j)
                    continue

                if j > i + self.max_duplicate_dist:
                    break

                d = pairwise_distances[i][j]
                if d > self.duplicate_threshold:
                    break

                neighbor_dist = []
                for shift in range(-self.duplicate_neighborhood, self.duplicate_neighborhood + 1):
                    shifted_i, shifted_j = i + shift, j + shift
                    if shifted_i < 0 or shifted_i >= len(traj):
                        continue
                    if shifted_j < 0 or shifted_j >= len(traj):
                        continue

                    neighbor_dist.append(pairwise_distances[i][shifted_j])
                    neighbor_dist.append(pairwise_distances[shifted_i][j])

                if np.percentile(neighbor_dist, 75) < self.duplicate_threshold:
                    log.info('Duplicate landmark frames %d-%d', i, j)
                    to_delete.add(j)
                else:
                    break

        log.debug('Removing duplicate frames %r from trajectory...', sorted(list(to_delete)))

        trajectory_class = traj.__class__
        new_trajectory = trajectory_class(traj.env_idx)

        for i in range(len(traj)):
            if i not in to_delete:
                new_trajectory.add_frame(traj, i)

        return new_trajectory

    @staticmethod
    def _add_simple_path_to_map(m, traj, node_idx):
        obs, infos = traj.obs, traj.infos
        curr_trajectory_idx = m.num_trajectories

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
            nodes[idx]['traj_idx'] = curr_trajectory_idx
            nodes[idx]['frame_idx'] = i

        m.frame_to_node_idx[curr_trajectory_idx] = node_idx
        m.num_trajectories += 1

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

                if j in m.graph[i]:
                    # there's already an edge between i and j
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
                distance_percentile = np.percentile(neighbors_dist, 75)
                shorcut_risk = distance_percentile  # closer to 0 = better

                # calculate ground-truth distance (purely for diagnostic purposes)
                xi = nodes[i]['info']['pos']['agent_x']
                yi = nodes[i]['info']['pos']['agent_y']
                xj = nodes[j]['info']['pos']['agent_x']
                yj = nodes[j]['info']['pos']['agent_y']
                gt_dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

                # Visualize false loop closures for debugging
                # if gt_dist > 50 and shorcut_risk < 0.1:
                #     log.info('Showing neighborhoods for shortcut %d-%d', i, j)
                #     import cv2
                #     obs_t1, obs_t2 = [], []
                #     for shift in range(-shortcut_window, shortcut_window + 1):
                #         shifted_i, shifted_j = i + shift, j + shift
                #         if shifted_i < 0 or shifted_i >= m.num_landmarks():
                #             continue
                #         if shifted_j < 0 or shifted_j >= m.num_landmarks():
                #             continue
                #
                #         shifted_i_traj_idx = nodes[shifted_i].get('traj_idx', 0)
                #         shifted_j_traj_idx = nodes[shifted_j].get('traj_idx', 0)
                #         if shifted_i_traj_idx != i_traj_idx or shifted_j_traj_idx != j_traj_idx:
                #             continue
                #
                #         obs_t1.append(nodes[shifted_i]['obs'])
                #         obs_t2.append(nodes[shifted_j]['obs'])
                #
                #     assert len(obs_t1) == len(obs_t2)
                #     for obs_i in range(len(obs_t1)):
                #         log.warning('Dist: %.3f', neighbors_dist[obs_i])
                #         cv2.imshow('t1', cv2.resize(cv2.cvtColor(obs_t1[obs_i], cv2.COLOR_RGB2BGR), (210, 210)))
                #         cv2.imshow('t2', cv2.resize(cv2.cvtColor(obs_t2[obs_i], cv2.COLOR_RGB2BGR), (210, 210)))
                #         cv2.waitKey()
                #     cv2.waitKey()

                shortcut = shorcut_risk, i, j, d, gt_dist
                assert i < j
                shortcut_candidates.append(shortcut)

        return shortcut_candidates

    @staticmethod
    def remove_shortcuts(m):
        to_remove = []

        for e in m.graph.edges(data=True):
            i1, i2, data = e
            if data.get('loop_closure', False):
                to_remove.append((i1, i2))

        log.info('Removing shortcut edges %r', to_remove)
        m.remove_edges_from(to_remove)

    def _add_shortcuts(self, m, pairwise_distances):
        self.remove_shortcuts(m)  # first - remove all existing shortcuts

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

    def add_trajectory_to_dense_map(self, existing_map, traj):
        t = Timing()

        m = existing_map
        m.new_episode()  # just in case

        node_idx = [-1] * len(traj)  # index map from trajectory frame to graph node idx
        node_idx[0] = 0  # first observation is always the same (we start from the same initial state)

        with t.timeit('create_initial_map'):
            self._add_simple_path_to_map(m, traj, node_idx)

        # precalculate feature vectors for the distance network
        with t.timeit('cache_feature_vectors'):
            all_observations = [m.get_observation(node) for node in m.graph.nodes]
            obs_embeddings = self._calc_embeddings(all_observations)

        with t.add_time('pairwise_distances'):
            pairwise_distances = self._calc_pairwise_distances(obs_embeddings)

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

    @staticmethod
    def calc_distances_to_landmarks(sparse_map, dense_map):
        landmarks = list(sparse_map.graph.nodes)

        for landmark in landmarks:
            node_data = sparse_map.graph.nodes[landmark]
            traj_idx = node_data.get('traj_idx', 0)
            frame_idx = node_data.get('frame_idx', 0)

            dense_map_landmark = dense_map.frame_to_node_idx[traj_idx][frame_idx]
            path = dense_map.get_path(0, dense_map_landmark, edge_weight=edge_weight)
            sparse_map.graph.nodes[landmark]['distance'] = len(path)

    @staticmethod
    def sieve_landmarks_by_distance(sparse_map, max_distance=1e9):
        landmarks = list(sparse_map.graph.nodes)
        sieved = [l for l in landmarks if sparse_map.graph.nodes[l]['distance'] <= max_distance]
        return sieved
