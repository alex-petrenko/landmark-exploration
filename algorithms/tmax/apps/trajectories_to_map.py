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
    # this is just a test
    m = init_map()
    map_builder = MapBuilder(agent, agent.distance.obs_encoder)
    for t in trajectories:
        m.new_episode()
        is_frame_a_landmark = map_builder.add_trajectory_to_sparse_map(m, t)
        landmark_frames = np.nonzero(is_frame_a_landmark)
        log.info('Landmark frames %r', landmark_frames)
    sparse_map_dir = ensure_dir_exists(join(trajectories_dir, 'sparse_map'))
    m.save_checkpoint(sparse_map_dir, map_img=map_img, coord_limits=coord_limits, verbose=True, is_sparse=True)


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

    def init_map():
        return TopologicalMap(
            trajectories[0].obs[0],
            directed_graph=False,
            initial_info=trajectories[0].infos[0],
            verbose=True,
        )

    test_sparse_map = False
    if test_sparse_map:
        trajectories_to_sparse_map(init_map, trajectories, trajectories_dir, agent, map_img, coord_limits)

    m = init_map()
    map_builder = MapBuilder(agent, agent.distance.obs_encoder)

    for t in trajectories:
        m = map_builder.add_trajectory_to_dense_map(m, t)

    dense_map_dir = ensure_dir_exists(join(trajectories_dir, 'dense_map'))
    m.save_checkpoint(dense_map_dir, map_img=map_img, coord_limits=coord_limits, verbose=True)

    agent.finalize()
    return 0


def main():
    args, params = parse_args_tmax(AgentTMAX.Params)
    status = trajectory_to_map(params, args.env)
    return status


if __name__ == '__main__':
    sys.exit(main())
