import glob
import math
import pickle
import random
import sys
from os.path import join

import numpy as np

from algorithms.topological_maps.map_builder import MapBuilder
from algorithms.utils.observation_encoder import ObservationEncoder
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
from algorithms.topological_maps.topological_map import TopologicalMap, hash_observation
from algorithms.utils.trajectory import Trajectory
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

    trajectories = []
    for i, trajectory_dir in enumerate(all_trajectories):
        with open(join(trajectory_dir, 'trajectory.pickle'), 'rb') as traj_file:
            traj = Trajectory(i)
            traj.__dict__.update(pickle.load(traj_file))
            trajectories.append(traj)

    m = TopologicalMap(
        trajectories[0].obs[0],
        directed_graph=False,
        initial_info=trajectories[0].infos[0],
        verbose=True,
    )

    map_builder = MapBuilder(agent, agent.distance.obs_encoder)

    for t in trajectories:
        m = map_builder.add_trajectory_to_map(m, t)

    combined_trajectory_dir = ensure_dir_exists(join(trajectories_dir, 'combined_trajectory'))
    m.save_checkpoint(combined_trajectory_dir, map_img=map_img, coord_limits=coord_limits, verbose=True)

    agent.finalize()
    return 0


def main():
    args, params = parse_args_tmax(AgentTMAX.Params)
    status = trajectory_to_map(params, args.env)
    return status


if __name__ == '__main__':
    sys.exit(main())
