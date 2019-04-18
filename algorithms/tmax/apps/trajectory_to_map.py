import glob
import math
import pickle
import random
import sys
from os.path import join

import numpy as np

from algorithms.reachability.observation_encoder import ObservationEncoder
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
from algorithms.topological_maps.topological_map import TopologicalMap, hash_observation
from utils.envs.envs import create_env
from utils.envs.generate_env_map import generate_env_map
from utils.timing import Timing
from utils.utils import ensure_dir_exists, AttrDict, log


def calc_pairwise_distances(m, agent):
    num_landmarks = m.num_landmarks()

    pairwise_distances = np.empty([num_landmarks, num_landmarks], np.float32)
    all_encoded_obs = [data['embedding'] for node, data in m.graph.nodes.data()]

    for i in range(num_landmarks):
        if i % 10 == 0:
            log.debug('Pairwise distances for %05d...', i)

        curr_obs = [all_encoded_obs[i]] * num_landmarks
        d = agent.curiosity.reachability.distances(agent.session, curr_obs, all_encoded_obs)
        pairwise_distances[i, :] = d

    for i in range(num_landmarks):
        for j in range(i + 1, num_landmarks):
            d = (pairwise_distances[i][j] + pairwise_distances[j][i]) * 0.5
            pairwise_distances[i][j] = pairwise_distances[j][i] = d

    return pairwise_distances


def shortcuts_sptm(pairwise_distances, num_landmarks, min_shortcut_dist, shortcut_window, m):
    """Loop closure generation inspired by SPTM."""
    shortcuts = []
    for i in range(num_landmarks):
        if i % 10 == 0:
            log.debug('Checking loop closures for %05d...', i)

        for j in range(i + min_shortcut_dist, num_landmarks):
            d = pairwise_distances[i][j]
            if d > 0.1:
                continue

            neighbors_dist = []

            for shift in range(-shortcut_window, shortcut_window + 1):
                shifted_i = i + shift
                if shifted_i < 0 or shifted_i >= num_landmarks:
                    continue

                shifted_j = j + shift
                if shifted_j < 0 or shifted_j >= num_landmarks:
                    continue

                neighbors_dist.append(pairwise_distances[shifted_i][shifted_j])

            shortcut_quality = np.median(neighbors_dist)  # closer to 0 = better

            shortcut = shortcut_quality, i, j, d
            assert i < j
            shortcuts.append(shortcut)

    return shortcuts


def shortcuts_distance(pairwise_distances, num_landmarks, min_shortcut_dist, shortcut_window, m):
    shortcuts = []
    threshold = 0.1

    for i in range(num_landmarks):
        if i % 10 == 0:
            log.debug('Checking loop closures for %05d...', i)

        for j in range(i + min_shortcut_dist, num_landmarks):
            neighbors_dist = []

            d = pairwise_distances[i][j]
            if d > threshold:
                continue

            for shift in range(-shortcut_window, shortcut_window + 1):
                shifted_i, shifted_j = i + shift, j + shift
                if shifted_i < 0 or shifted_i >= num_landmarks:
                    continue
                if shifted_j < 0 or shifted_j >= num_landmarks:
                    continue

                neighbors_dist.append(pairwise_distances[shifted_i][shifted_j])

            distance_percentile = np.percentile(neighbors_dist, 60)

            nodes = m.graph.nodes
            xi = nodes[i]['info']['pos']['agent_x']
            yi = nodes[i]['info']['pos']['agent_y']
            xj = nodes[j]['info']['pos']['agent_x']
            yj = nodes[j]['info']['pos']['agent_y']
            coord_dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)

            shortcut_quality = distance_percentile
            shortcut = shortcut_quality, i, j, d, coord_dist
            assert i < j
            shortcuts.append(shortcut)

    return shortcuts


def trajectory_to_map(params, env_id):
    def make_env_func():
        e = create_env(env_id)
        e.seed(0)
        return e

    agent = AgentTMAX(make_env_func, params)
    agent.initialize()

    map_img, coord_limits = generate_env_map(make_env_func)

    experiment_dir = params.experiment_dir()
    trajectories_dir = ensure_dir_exists(join(experiment_dir, '.trajectories'))

    prefix = 'traj_'
    all_trajectories = glob.glob(f'{trajectories_dir}/{prefix}*')
    all_trajectories.sort()

    trajectory_dir = all_trajectories[-1]
    with open(join(trajectory_dir, 'trajectory.pickle'), 'rb') as traj_file:
        traj = pickle.load(traj_file)

    m = TopologicalMap(traj[0]['obs'], directed_graph=False, initial_info=traj[0]['info'], verbose=True)

    t = Timing()

    with t.timeit('create_initial_map'):
        for i in range(len(traj)):
            frame = AttrDict(traj[i])
            obs, info = frame.obs, frame.info

            if i > 0:
                idx = m.add_landmark(obs, info, update_curr_landmark=True)
                assert idx == i

            m.graph.nodes[i]['info'] = info
            m.graph.nodes[i]['embedding'] = None

    # precalculate feature vectors for the distances network
    with t.timeit('cache_feature_vectors'):
        encoder = ObservationEncoder(encode_func=agent.curiosity.reachability.encode_observation)
        obs_to_encode = [frame['obs'] for frame in traj]
        obs_hashes = [hash_observation(o) for o in obs_to_encode]

        encoder.encode(agent.session, obs_to_encode)

        assert len(traj) == len(obs_hashes)

        for i in range(len(traj)):
            obs_hash = obs_hashes[i]
            m.graph.nodes[i]['embedding'] = encoder.encoded_obs[obs_hash]

    # map generation parameters
    max_duplicate_dist = 20
    duplicate_neighborhood = 5
    duplicate_threshold = 0.1

    min_shortcut_dist = 5
    shortcut_window = 10
    shortcuts_to_keep = int(0.3 * m.num_landmarks())

    with t.add_time('pairwise_distances'):
        pairwise_distances = calc_pairwise_distances(m, agent)

    with t.timeit('sparsify'):
        to_delete = []
        for i in range(m.num_landmarks()):
            if i in to_delete:
                continue

            for j in range(i + 1, min(m.num_landmarks(), i + 1 + max_duplicate_dist)):
                if j in to_delete:
                    continue

                neighbor_dist = []

                d = pairwise_distances[i][j]
                if d > duplicate_threshold:
                    continue

                for shift in range(-duplicate_neighborhood, duplicate_neighborhood + 1):
                    shifted_i, shifted_j = i + shift, j + shift
                    if shifted_i < 0 or shifted_i >= m.num_landmarks():
                        continue
                    if shifted_j < 0 or shifted_j >= m.num_landmarks():
                        continue

                    neighbor_dist.append(pairwise_distances[i][shifted_j])
                    neighbor_dist.append(pairwise_distances[shifted_i][j])

                if np.percentile(neighbor_dist, 50) < duplicate_threshold:
                    log.info('Duplicate landmark %d-%d', i, j)
                    to_delete.append(j)

        new_map = TopologicalMap(traj[0]['obs'], directed_graph=False, initial_info=traj[0]['info'], verbose=True)
        for node, data in m.graph.nodes.data():
            if node in to_delete:
                continue

            idx = 0
            if node > 0:
                idx = new_map.add_landmark(data['obs'], data['info'], update_curr_landmark=True)

            for key, value in data.items():
                new_map.graph.nodes[idx][key] = value

        m = new_map

    with t.add_time('pairwise_distances'):
        pairwise_distances = calc_pairwise_distances(m, agent)

    with t.timeit('loop_closures'):
        shortcuts = shortcuts_distance(pairwise_distances, m.num_landmarks(), min_shortcut_dist, shortcut_window, m)

        random.shuffle(shortcuts)
        shortcut_qualities = [s[0] for s in shortcuts]

        keep = min(shortcuts_to_keep, len(shortcuts))
        percentile = (keep / len(shortcuts)) * 100
        max_distance_to_keep = np.percentile(shortcut_qualities, percentile)
        max_distance_to_keep = min(max_distance_to_keep, 0.15)

        log.debug('Keep shortcuts with distance <= %.3f...', max_distance_to_keep)
        shortcuts = [s for s in shortcuts if s[0] <= max_distance_to_keep][:keep]
        shortcuts.sort(reverse=True)
        shortcuts.sort(key=lambda x: x[-1], reverse=True)

        log.debug('Kept %d shortcuts: %r...', len(shortcuts), shortcuts[:5])

    with t.timeit('modifying_map'):
        for shortcut in shortcuts:
            quality, i1, i2, d, coord_dist = shortcut
            m.add_edge(i1, i2, loop_closure=True)

    with t.timeit('save_map'):
        m.save_checkpoint(trajectory_dir, map_img=map_img, coord_limits=coord_limits, verbose=True)

    log.debug('Timing: %s', t)

    agent.finalize()
    return 0


def main():
    args, params = parse_args_tmax(AgentTMAX.Params)
    status = trajectory_to_map(params, args.env)
    return status


if __name__ == '__main__':
    sys.exit(main())
