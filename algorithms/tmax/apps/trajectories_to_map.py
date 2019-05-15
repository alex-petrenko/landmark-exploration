import copy
import glob
import pickle
import sys
from os.path import join

import numpy as np

from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
from algorithms.topological_maps.map_builder import MapBuilder
from algorithms.topological_maps.topological_map import TopologicalMap
from algorithms.utils.trajectory import Trajectory
from utils.envs.envs import create_env
from utils.envs.generate_env_map import generate_env_map
from utils.utils import ensure_dir_exists, log


def trajectories_to_sparse_map(init_map, trajectories, trajectories_dir, agent, map_img, coord_limits):
    """This is just a test."""
    m = init_map()
    map_builder = MapBuilder(agent)
    for t in trajectories:
        m.new_episode()
        is_frame_a_landmark = map_builder.add_trajectory_to_sparse_map_fixed_landmarks(m, t)
        landmark_frames = np.nonzero(is_frame_a_landmark)
        log.info('Landmark frames %r', landmark_frames)
    sparse_map_dir = ensure_dir_exists(join(trajectories_dir, 'sparse_map'))
    m.save_checkpoint(sparse_map_dir, map_img=map_img, coord_limits=coord_limits, verbose=True, is_sparse=True)
    return m


def pick_best_trajectory(init_map, agent, trajectories):
    """This is just a test."""
    m = init_map()
    map_builder = MapBuilder(agent)
    map_builder.add_trajectory_to_sparse_map_fixed_landmarks(m, trajectories[0])

    # noinspection PyProtectedMember
    best_t_idx, best_t_dist = agent.tmax_mgr._pick_best_exploration_trajectory_avg_distance(agent, trajectories, m)
    log.info('Best traj index %d with dist %.3f', best_t_idx, best_t_dist)


def trajectory_to_map(params, env_id):
    def make_env_func():
        e = create_env(env_id)
        e.seed(0)
        return e

    params.num_envs = 1
    params.with_timer = False
    agent = AgentTMAX(make_env_func, params)
    agent.initialize()

    map_img, coord_limits = generate_env_map(make_env_func)

    experiment_dir = params.experiment_dir()
    trajectories_dir = ensure_dir_exists(join(experiment_dir, '.trajectories'))

    if params.persistent_map_checkpoint is None:
        prefix = 'traj_'
        all_trajectories = glob.glob(f'{trajectories_dir}/{prefix}*')
        all_trajectories.sort()

        trajectories = []
        for i, trajectory_dir in enumerate(all_trajectories):
            with open(join(trajectory_dir, 'trajectory.pickle'), 'rb') as traj_file:
                traj = Trajectory(i)
                traj.__dict__.update(pickle.load(traj_file))
                trajectories.append(traj)
    else:
        loaded_persistent_map = TopologicalMap.create_empty()
        loaded_persistent_map.maybe_load_checkpoint(params.persistent_map_checkpoint)

        num_trajectories = loaded_persistent_map.num_trajectories
        trajectories = [Trajectory(i) for i in range(num_trajectories)]

        zero_frame = loaded_persistent_map.graph.nodes[0]
        for i in range(1, num_trajectories):
            trajectories[i].add(zero_frame['obs'], -1, zero_frame['info'])

        for node in loaded_persistent_map.graph.nodes(data=True):
            node_idx, d = node
            trajectories[d['traj_idx']].add(d['obs'], -1, d['info'])

        log.info('Loaded %d trajectories from the map', num_trajectories)
        log.info('Trajectory lengths %r', [len(t) for t in trajectories])

    def init_map():
        return TopologicalMap(
            trajectories[0].obs[0],
            directed_graph=False,
            initial_info=trajectories[0].infos[0],
        )

    map_builder = MapBuilder(agent)
    # trajectories = [map_builder.sparsify_trajectory(t) for t in trajectories]  # TODO

    sparse_map = trajectories_to_sparse_map(
        init_map, trajectories, trajectories_dir, agent, map_img, coord_limits,
    )

    test_pick_best_trajectory = True
    if test_pick_best_trajectory:
        pick_best_trajectory(init_map, agent, copy.deepcopy(trajectories))

    m = init_map()

    for i, t in enumerate(trajectories):
        m = map_builder.add_trajectory_to_dense_map(m, t)

    map_builder.calc_distances_to_landmarks(sparse_map, m)
    map_builder.sieve_landmarks_by_distance(sparse_map)

    dense_map_dir = ensure_dir_exists(join(trajectories_dir, 'dense_map'))
    m.save_checkpoint(dense_map_dir, map_img=map_img, coord_limits=coord_limits, verbose=True)

    # check if landmark correspondence between dense and sparse map is correct
    for node, data in sparse_map.graph.nodes.data():
        traj_idx = data['traj_idx']
        frame_idx = data['frame_idx']

        dense_map_landmark = m.frame_to_node_idx[traj_idx][frame_idx]
        log.info('Sparse map node %d corresponds to dense map node %d', node, dense_map_landmark)
        log.info('Sparse map node %d distance %d', node, data['distance'])

        obs_sparse = sparse_map.get_observation(node)
        obs_dense = m.get_observation(dense_map_landmark)

        assert np.array_equal(obs_sparse, obs_dense)

        show_landmarks = False
        if show_landmarks:
            import cv2
            cv2.imshow('sparse', obs_sparse)
            cv2.imshow('dense', obs_dense)
            cv2.waitKey()

    agent.finalize()
    return 0


def main():
    args, params = parse_args_tmax(AgentTMAX.Params)
    status = trajectory_to_map(params, args.env)
    return status


if __name__ == '__main__':
    sys.exit(main())
