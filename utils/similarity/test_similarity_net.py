import sys

import tensorflow as tf

from algorithms.env_wrappers import get_observation_space
from utils.envs.envs import create_env
from utils.similarity.similarity_net import SimilarityNetwork
from utils.utils import model_dir, log, experiment_dir


def test(env_id):
    def make_env_func():
        return create_env(env_id)

    env = make_env_func()  # Creating env for now to get observations space, etc

    obs_shape = list(get_observation_space(env).shape)
    input_shape = [None] + obs_shape  # add batch dimension

    ph_ob1 = tf.placeholder(tf.float32, shape=input_shape)
    ph_ob2 = tf.placeholder(tf.float32, shape=input_shape)

    sim_net = SimilarityNetwork(ph_ob1, ph_ob2, "similarity")

    return 0


def init():
    gpu_options = tf.GPUOptions()

    config = tf.ConfigProto(
        device_count={'GPU': 100},
        gpu_options=gpu_options,
        log_device_placement=False,
    )

    session = tf.Session(config=config)

    checkpoint_dir = model_dir(experiment_dir('test_similarity_net'))
    saver = tf.train.Saver(max_to_keep=3)

    try:
        saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir))
    except ValueError:
        log.info('Didn\'t find a valid restore point, start from scratch')
        session.run(tf.global_variables_initializer())


def main():
    """Script entry point."""
    return test()


if __name__ == '__main__':
    sys.exit(main())
