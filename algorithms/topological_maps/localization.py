import math

import numpy as np

from algorithms.topological_maps.topological_map import hash_observation
from utils.timing import Timing
from utils.utils import log, min_with_idx


class Localizer:
    def __init__(self, params, obs_encoder, verbose=False):
        self._verbose = verbose

        self.params = params
        self.new_landmark_threshold = self.params.new_landmark_threshold
        self.loop_closure_threshold = self.params.loop_closure_threshold

        # noise-filtering parameter, how many frames we need to wait before we change localization
        self.localize_frames = 4

        self.num_envs = self.params.num_envs

        self.obs_encoder = obs_encoder

    def _log_verbose(self, s, *args):
        if self._verbose:
            log.debug(s, *args)

    def _log_distances(self, env_i, neighbor_indices, distance):
        """Optional diagnostic logging."""
        log_reachability = True
        if self._verbose and log_reachability:
            neighbor_distance = {}
            for i, neighbor_idx in enumerate(neighbor_indices):
                neighbor_distance[neighbor_idx] = '{:.3f}'.format(distance[i])
            self._log_verbose('Env %d distance: %r', env_i, neighbor_distance)

    def localize(self, session, obs, info, maps, reachability, timing=None):
        if timing is None:
            timing = Timing()

        bonuses = np.zeros([self.num_envs])

        with timing.add_time('encode_obs'):
            obs_hashes = [hash_observation(o) for o in obs]
            self.obs_encoder.encode(session, obs, obs_hashes)
            obs_encoded = [self.obs_encoder.encoded_obs[obs_hash] for obs_hash in obs_hashes]

        # create a batch of all neighborhood observations from all envs for fast processing on GPU
        neighborhood_obs, neighborhood_hashes, current_obs = [], [], []
        for env_i, m in enumerate(maps):
            neighbor_indices = m.neighborhood()
            neighborhood_obs.extend([m.get_observation(i) for i in neighbor_indices])
            neighborhood_hashes.extend([m.get_hash(i) for i in neighbor_indices])
            current_obs.extend([obs_encoded[env_i]] * len(neighbor_indices))

        assert len(neighborhood_obs) == len(current_obs)
        assert len(neighborhood_obs) == len(neighborhood_hashes)

        with timing.add_time('neighbor_dist'):
            self.obs_encoder.encode(session, neighborhood_obs, neighborhood_hashes)
            neighborhood_encoded = [self.obs_encoder.encoded_obs[h] for h in neighborhood_hashes]

            # calculate reachability for all neighborhoods in all envs
            distances = reachability.distances(session, neighborhood_encoded, current_obs)

        new_landmark_candidates = []
        closest_landmark_idx = [-1] * self.num_envs

        j = 0
        for env_i, m in enumerate(maps):
            neighbor_indices = m.neighborhood()
            j_next = j + len(neighbor_indices)
            distance = distances[j:j_next]

            self._log_distances(env_i, neighbor_indices, distance)

            # check if we're far enough from all landmarks in the neighborhood
            min_d, min_d_idx = min_with_idx(distance)
            closest_landmark_idx[env_i] = neighbor_indices[min_d_idx]

            if min_d >= self.new_landmark_threshold:
                # we're far enough from all obs in the neighborhood, might have found something new!
                new_landmark_candidates.append(env_i)
            else:
                # we're still sufficiently close to our neighborhood, but maybe "current landmark" has changed
                m.new_landmark_candidate_frames = 0

                # crude localization
                if all(lm == closest_landmark_idx[env_i] for lm in m.closest_landmarks[-self.localize_frames:]):
                    if closest_landmark_idx[env_i] != m.curr_landmark_idx:
                        m.set_curr_landmark(closest_landmark_idx[env_i])

            j = j_next

        del neighborhood_obs
        del neighborhood_hashes
        del neighborhood_encoded
        del current_obs

        # Agents in some environments discovered landmarks that are far away from all landmarks in the immediate
        # vicinity. There are two possibilities:
        # 1) A new landmark should be created and added to the graph
        # 2) We're close to some other vertex in the graph - we've found a "loop closure", a new edge in a graph

        non_neighborhood_obs, non_neighborhood_hashes = [], []
        non_neighborhoods = {}
        current_obs = []
        for env_i in new_landmark_candidates:
            m = maps[env_i]
            non_neighbor_indices = m.curr_non_neighbors()
            non_neighborhoods[env_i] = non_neighbor_indices
            non_neighborhood_obs.extend([m.get_observation(i) for i in non_neighbor_indices])
            non_neighborhood_hashes.extend([m.get_hash(i) for i in non_neighbor_indices])
            current_obs.extend([obs_encoded[env_i]] * len(non_neighbor_indices))

        assert len(non_neighborhood_obs) == len(current_obs)

        with timing.add_time('non_neigh'):
            # calculate reachability for all non-neighbors
            distances = []
            batch_size = 1024
            for i in range(0, len(non_neighborhood_obs), batch_size):
                start, end = i, i + batch_size
                self.obs_encoder.encode(session, non_neighborhood_obs[start:end], non_neighborhood_hashes[start:end])
                non_neighborhood_encoded = [self.obs_encoder.encoded_obs[h] for h in non_neighborhood_hashes]

                distances_batch = reachability.distances(
                    session, non_neighborhood_encoded[start:end], current_obs[start:end],
                )
                distances.extend(distances_batch)

        j = 0
        for env_i in new_landmark_candidates:
            m = maps[env_i]
            non_neighbor_indices = non_neighborhoods[env_i]
            j_next = j + len(non_neighbor_indices)
            distance = distances[j:j_next]

            min_d, min_d_idx = math.inf, math.inf
            if len(distance) > 0:
                min_d, min_d_idx = min_with_idx(distance)

            if min_d < self.loop_closure_threshold:
                # current observation is close to some other landmark, "close the loop" by creating a new edge
                m.new_landmark_candidate_frames = 0

                closest_landmark_idx[env_i] = non_neighbor_indices[min_d_idx]

                # crude localization
                if all(lm == closest_landmark_idx[env_i] for lm in m.closest_landmarks[-self.localize_frames:]):
                    m.set_curr_landmark(closest_landmark_idx[env_i])

                    bonuses[env_i] += self.params.map_expansion_reward  # we found a new edge! Cool!
            else:
                # vertex is relatively far away from all vertices in the graph, we've found a new landmark!
                if m.new_landmark_candidate_frames >= self.localize_frames:
                    new_landmark_idx = m.add_landmark(obs[env_i], info[env_i])
                    m.set_curr_landmark(new_landmark_idx)

                    closest_landmark_idx[env_i] = new_landmark_idx
                    m.new_landmark_candidate_frames = 0
                else:
                    m.new_landmark_candidate_frames += 1

            j = j_next

        # update localization info
        for env_i in range(self.num_envs):
            assert closest_landmark_idx[env_i] >= 0
            maps[env_i].closest_landmarks.append(closest_landmark_idx[env_i])

        return bonuses
