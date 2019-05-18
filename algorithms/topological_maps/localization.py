import math

from algorithms.topological_maps.topological_map import hash_observation
from utils.timing import Timing
from utils.utils import log, min_with_idx


class Localizer:
    def __init__(self, params, verbose=False):
        self._verbose = verbose

        self.params = params
        self.new_landmark_threshold = self.params.new_landmark_threshold
        self.loop_closure_threshold = self.params.loop_closure_threshold

        # noise-filtering parameter, how many frames we need to wait before we change localization
        self.localize_frames = 3

    def _log_verbose(self, s, *args):
        if self._verbose:
            log.debug(s, *args)

    def _log_distances(self, env_i, neighbor_indices, distance):
        """Optional diagnostic logging."""
        enable_logging = True
        if self._verbose and enable_logging:
            neighbor_distance = {}
            for i, neighbor_idx in enumerate(neighbor_indices):
                neighbor_distance[neighbor_idx] = '{:.3f}'.format(distance[i])
            self._log_verbose('Env %d distance: %r', env_i, neighbor_distance)

    def localize(
            self,
            session, obs, info, maps, distance_net, frames=None, on_new_landmark=None, on_new_edge=None, timing=None,
    ):
        num_envs = len(obs)
        closest_landmark_idx = [-1] * num_envs
        # closest distance to the landmark in the existing graph (excluding new landmarks)
        closest_landmark_dist = [math.inf] * num_envs

        if all(m is None for m in maps):
            return closest_landmark_dist

        if timing is None:
            timing = Timing()

        # create a batch of all neighborhood observations from all envs for fast processing on GPU
        neighborhood_obs, neighborhood_hashes, current_obs, current_obs_hashes = [], [], [], []
        neighborhood_infos, current_infos = [], []
        total_num_neighbors = 0
        neighborhood_sizes = [0] * len(maps)
        for env_i, m in enumerate(maps):
            if m is None:
                continue

            neighbor_indices = m.neighborhood()
            neighborhood_sizes[env_i] = len(neighbor_indices)
            neighborhood_obs.extend([m.get_observation(i) for i in neighbor_indices])
            neighborhood_infos.extend([m.get_info(i) for i in neighbor_indices])
            neighborhood_hashes.extend([m.get_hash(i) for i in neighbor_indices])
            current_obs.extend([obs[env_i]] * len(neighbor_indices))
            current_obs_hashes.extend([hash_observation(obs[env_i])] * len(neighbor_indices))
            current_infos.extend([info[env_i]] * len(neighbor_indices))
            total_num_neighbors += len(neighbor_indices)

        assert len(neighborhood_obs) == len(current_obs)
        assert len(neighborhood_obs) == len(neighborhood_hashes)
        assert len(current_obs) == total_num_neighbors
        assert len(neighborhood_infos) == len(current_infos)

        with timing.add_time('neighbor_dist'):
            distances = distance_net.distances_from_obs(
                session,
                obs_first=neighborhood_obs, obs_second=current_obs,
                hashes_first=neighborhood_hashes, hashes_second=current_obs_hashes,
                infos_first=neighborhood_infos, infos_second=current_infos,
            )

        assert len(distances) == total_num_neighbors

        new_landmark_candidates = []

        j = 0
        for env_i, m in enumerate(maps):
            if m is None:
                continue

            neighbor_indices = m.neighborhood()
            j_next = j + len(neighbor_indices)
            distance = distances[j:j_next]

            if len(neighbor_indices) != neighborhood_sizes[env_i]:
                log.warning(
                    'For env %d neighbors size expected %d, actual %d',
                    env_i, neighborhood_sizes[env_i], len(neighbor_indices),
                )

            assert len(neighbor_indices) == neighborhood_sizes[env_i]

            self._log_distances(env_i, neighbor_indices, distance)

            j = j_next

            # check if we're far enough from all landmarks in the neighborhood
            min_d, min_d_idx = min_with_idx(distance)
            closest_landmark_idx[env_i] = neighbor_indices[min_d_idx]
            closest_landmark_dist[env_i] = min_d

            if min_d >= self.new_landmark_threshold:
                # we're far enough from all obs in the neighborhood, might have found something new!
                new_landmark_candidates.append(env_i)
            else:
                # we're still sufficiently close to our neighborhood, but maybe "current landmark" has changed
                m.new_landmark_candidate_frames = 0
                m.loop_closure_candidate_frames = 0

                # crude localization
                if all(lm == closest_landmark_idx[env_i] for lm in m.closest_landmarks[-self.localize_frames:]):
                    if closest_landmark_idx[env_i] != m.curr_landmark_idx:
                        m.set_curr_landmark(closest_landmark_idx[env_i])

        del neighborhood_obs
        del neighborhood_infos
        del neighborhood_hashes
        del current_obs
        del current_infos

        # Agents in some environments discovered landmarks that are far away from all landmarks in the immediate
        # vicinity. There are two possibilities:
        # 1) A new landmark should be created and added to the graph
        # 2) We're close to some other vertex in the graph - we've found a "loop closure", a new edge in a graph

        non_neighborhood_obs, non_neighborhood_hashes = [], []
        non_neighborhoods = {}
        current_obs, current_obs_hashes = [], []
        non_neighborhood_infos, current_infos = [], []
        for env_i in new_landmark_candidates:
            m = maps[env_i]
            if m is None:
                continue

            non_neighbor_indices = m.curr_non_neighbors()
            non_neighborhoods[env_i] = non_neighbor_indices
            non_neighborhood_obs.extend([m.get_observation(i) for i in non_neighbor_indices])
            non_neighborhood_infos.extend([m.get_info(i) for i in non_neighbor_indices])
            non_neighborhood_hashes.extend([m.get_hash(i) for i in non_neighbor_indices])
            current_obs.extend([obs[env_i]] * len(non_neighbor_indices))
            current_obs_hashes.extend([hash_observation(obs[env_i])] * len(non_neighbor_indices))
            current_infos.extend([info[env_i]] * len(non_neighbor_indices))

        assert len(non_neighborhood_obs) == len(current_obs)
        assert len(non_neighborhood_obs) == len(non_neighborhood_hashes)

        with timing.add_time('non_neigh'):
            # calculate distance for all non-neighbors
            distances = []
            batch_size = 1024
            for i in range(0, len(non_neighborhood_obs), batch_size):
                start, end = i, i + batch_size

                distances_batch = distance_net.distances_from_obs(
                    session,
                    obs_first=non_neighborhood_obs[start:end], obs_second=current_obs[start:end],
                    hashes_first=non_neighborhood_hashes[start:end], hashes_second=current_obs_hashes[start:end],
                    infos_first=non_neighborhood_infos[start:end], infos_second=current_infos[start:end],
                )
                distances.extend(distances_batch)

        j = 0
        for env_i in new_landmark_candidates:
            m = maps[env_i]
            if m is None:
                continue

            non_neighbor_indices = non_neighborhoods[env_i]
            j_next = j + len(non_neighbor_indices)
            distance = distances[j:j_next]
            j = j_next

            min_d, min_d_idx = math.inf, math.inf
            if len(distance) > 0:
                min_d, min_d_idx = min_with_idx(distance)
                closest_landmark_dist[env_i] = min(closest_landmark_dist[env_i], min_d)

            if min_d < self.loop_closure_threshold:
                # current observation is close to some other landmark, "close the loop" by creating a new edge
                m.new_landmark_candidate_frames = 0
                m.loop_closure_candidate_frames += 1

                closest_landmark_idx[env_i] = non_neighbor_indices[min_d_idx]

                # crude localization
                if m.loop_closure_candidate_frames >= self.localize_frames:
                    if all(lm == closest_landmark_idx[env_i] for lm in m.closest_landmarks[-self.localize_frames:]):
                        # we found a new edge! Cool!
                        m.loop_closure_candidate_frames = 0
                        m.set_curr_landmark(closest_landmark_idx[env_i])

                        if on_new_edge is not None:
                            on_new_edge(env_i)

            elif min_d >= self.new_landmark_threshold:
                m.loop_closure_candidate_frames = 0
                m.new_landmark_candidate_frames += 1

                # vertex is relatively far away from all vertices in the graph, we've found a new landmark!
                if m.new_landmark_candidate_frames >= self.localize_frames:
                    new_landmark_idx = m.add_landmark(obs[env_i], info[env_i], update_curr_landmark=True)

                    if frames is not None:
                        m.graph.nodes[new_landmark_idx]['added_at'] = frames[env_i]

                    closest_landmark_idx[env_i] = new_landmark_idx
                    m.new_landmark_candidate_frames = 0

                    if on_new_landmark is not None:
                        on_new_landmark(env_i, new_landmark_idx)
            else:
                m.new_landmark_candidate_frames = 0
                m.loop_closure_candidate_frames = 0

        # update localization info
        for env_i in range(num_envs):
            m = maps[env_i]
            if m is None:
                continue

            assert closest_landmark_idx[env_i] >= 0
            m.closest_landmarks.append(closest_landmark_idx[env_i])

        # # visualize "closest" landmark
        # import cv2
        # closest_lm = maps[0].closest_landmarks[-1]
        # closest_obs = maps[0].get_observation(closest_lm)
        # cv2.imshow('closest_obs', cv2.resize(cv2.cvtColor(closest_obs, cv2.COLOR_RGB2BGR), (420, 420)))
        # cv2.waitKey(1)

        return closest_landmark_dist
