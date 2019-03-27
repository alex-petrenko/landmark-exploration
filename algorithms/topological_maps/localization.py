import math

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
        self.localize_frames = 3

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

    def localize(self, session, obs, info, maps, reachability, on_new_landmark=None, on_new_edge=None, timing=None):
        closest_landmark_idx = [-1] * self.num_envs
        # closest distance to the landmark in the existing graph (excluding new landmarks)
        closest_landmark_dist = [math.inf] * self.num_envs

        if all(m is None for m in maps):
            return closest_landmark_dist

        if timing is None:
            timing = Timing()

        # calculate feature vectors for new observations
        with timing.add_time('encode_obs'):
            obs_hashes = [hash_observation(o) for o in obs]
            self.obs_encoder.encode(session, obs, obs_hashes)
            obs_encoded = [self.obs_encoder.encoded_obs[obs_hash] for obs_hash in obs_hashes]

        # create a batch of all neighborhood observations from all envs for fast processing on GPU
        neighborhood_obs, neighborhood_hashes, current_obs = [], [], []
        total_num_neighbors = 0
        neighborhood_sizes = [0] * len(maps)
        for env_i, m in enumerate(maps):
            if m is None:
                continue

            neighbor_indices = m.neighborhood()
            neighborhood_sizes[env_i] = len(neighbor_indices)
            neighborhood_obs.extend([m.get_observation(i) for i in neighbor_indices])
            neighborhood_hashes.extend([m.get_hash(i) for i in neighbor_indices])
            current_obs.extend([obs_encoded[env_i]] * len(neighbor_indices))
            total_num_neighbors += len(neighbor_indices)

        assert len(neighborhood_obs) == len(current_obs)
        assert len(neighborhood_obs) == len(neighborhood_hashes)
        assert len(current_obs) == total_num_neighbors

        with timing.add_time('neighbor_dist'):
            self.obs_encoder.encode(session, neighborhood_obs, neighborhood_hashes)
            neighborhood_encoded = [self.obs_encoder.encoded_obs[h] for h in neighborhood_hashes]

            # calculate reachability for all neighborhoods in all envs
            distances = reachability.distances(session, neighborhood_encoded, current_obs)

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

            if len(distance) <= 0:
                log.warning('Distance to neighbors array empty! Neighbors %r, j %d jn %d', m.neighborhood(), j, j_next)
                log.warning('Current landmark %d', m.curr_landmark_idx)
                map_sizes = [mp.num_landmarks() if mp is not None else None for mp in maps]
                log.warning('Map sizes %r', map_sizes)
                log.warning('Distances array size %d', len(distances))
                log.warning('Distances array %r', distances)

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

                # crude localization
                if all(lm == closest_landmark_idx[env_i] for lm in m.closest_landmarks[-self.localize_frames:]):
                    if closest_landmark_idx[env_i] != m.curr_landmark_idx:
                        m.set_curr_landmark(closest_landmark_idx[env_i])

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
            if m is None:
                continue

            non_neighbor_indices = m.curr_non_neighbors()
            non_neighborhoods[env_i] = non_neighbor_indices
            non_neighborhood_obs.extend([m.get_observation(i) for i in non_neighbor_indices])
            non_neighborhood_hashes.extend([m.get_hash(i) for i in non_neighbor_indices])
            current_obs.extend([obs_encoded[env_i]] * len(non_neighbor_indices))

        assert len(non_neighborhood_obs) == len(current_obs)
        assert len(non_neighborhood_obs) == len(non_neighborhood_hashes)

        with timing.add_time('non_neigh'):
            # calculate reachability for all non-neighbors
            distances = []
            batch_size = 1024
            for i in range(0, len(non_neighborhood_obs), batch_size):
                start, end = i, i + batch_size
                self.obs_encoder.encode(session, non_neighborhood_obs[start:end], non_neighborhood_hashes[start:end])
                non_neighborhood_encoded = [self.obs_encoder.encoded_obs[h] for h in non_neighborhood_hashes[start:end]]
                assert len(non_neighborhood_encoded)

                distances_batch = reachability.distances(session, non_neighborhood_encoded, current_obs[start:end])
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

                closest_landmark_idx[env_i] = non_neighbor_indices[min_d_idx]

                # crude localization
                if all(lm == closest_landmark_idx[env_i] for lm in m.closest_landmarks[-self.localize_frames:]):
                    # we found a new edge! Cool!
                    m.set_curr_landmark(closest_landmark_idx[env_i])
                    if on_new_edge is not None:
                        on_new_edge(env_i)
            else:
                # vertex is relatively far away from all vertices in the graph, we've found a new landmark!
                if m.new_landmark_candidate_frames >= self.localize_frames:
                    new_landmark_idx = m.add_landmark(obs[env_i], info[env_i])
                    m.set_curr_landmark(new_landmark_idx)

                    closest_landmark_idx[env_i] = new_landmark_idx
                    m.new_landmark_candidate_frames = 0

                    if on_new_landmark is not None:
                        on_new_landmark(env_i)
                else:
                    m.new_landmark_candidate_frames += 1

        # update localization info
        for env_i in range(self.num_envs):
            m = maps[env_i]
            if m is None:
                continue

            assert closest_landmark_idx[env_i] >= 0
            m.closest_landmarks.append(closest_landmark_idx[env_i])

        return closest_landmark_dist
