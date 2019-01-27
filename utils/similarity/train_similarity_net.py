import sys
from os.path import join

import numpy as np
import tensorflow as tf

from algorithms.env_wrappers import get_observation_space
from utils.envs.envs import create_env
from utils.similarity.similarity_net import SimilarityNetwork
from utils.utils import model_dir, log, experiment_dir, data_dir


def train(env_id):
    def make_env_func():
        return create_env(env_id)

    env = make_env_func()  # Creating env for now to get observations space, etc

    obs_shape = list(get_observation_space(env).shape)
    input_shape = [None] + obs_shape  # add batch dimension

    ph_ob1 = tf.placeholder(tf.float32, shape=input_shape)
    ph_ob2 = tf.placeholder(tf.float32, shape=input_shape)

    sim_net = SimilarityNetwork(ph_ob1, ph_ob2, "similarity")

    session = init()

    trajectories = load_trajectories()
    actions = trajectories['action']
    dones = trajectories['done']
    obs = trajectories['obs']
    rewards = trajectories['reward']

    print(actions.shape, dones.shape, obs.shape, rewards.shape)
    print(obs_shape)

    obs_input_shape = [1] + obs_shape
    ob1 = obs[0].reshape(obs_input_shape)
    ob2 = obs[1].reshape(obs_input_shape)
    # train
    print(sim_net.pred(session, ob1, ob2))

    session.close()

    return 0


def init():
    gpu_options = tf.GPUOptions()

    config = tf.ConfigProto(
        device_count={'GPU': 100},
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    session = tf.Session(config=config)

    checkpoint_dir = model_dir(experiment_dir('similarity_net'))
    saver = tf.train.Saver(max_to_keep=3)

    try:
        saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir))
    except ValueError:
        log.info('Didn\'t find a valid restore point, start from scratch')
        session.run(tf.global_variables_initializer())

    log.info('Initialized!')
    return session


def load_trajectories():
    traj_dir = join(data_dir(experiment_dir('doom_maze-random_v000')), 'trajectories')
    trajectories = np.load(join(traj_dir, 'ep_0000000_random_traj.npz'))

    return trajectories


def main():
    """Script entry point."""
    return train('doom_maze')


if __name__ == '__main__':
    sys.exit(main())
