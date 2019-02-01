import os
import shutil
from unittest import TestCase

import tensorflow as tf

from algorithms.agent import TrainStatus
from algorithms.tests.test_wrappers import TEST_ENV_NAME
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.enjoy_tmax import enjoy
from algorithms.tmax.reachability import ReachabilityNetwork
from algorithms.tmax.tmax_utils import parse_args_tmax
from algorithms.tmax.train_tmax import train
from utils.envs.doom.doom_utils import make_doom_env, env_by_name


class TestTMAX(TestCase):
    def test_tmax_train_run(self):
        test_dir_name = self.__class__.__name__

        args, params = parse_args_tmax(AgentTMAX.Params)
        params.experiments_root = test_dir_name
        params.num_envs = 16
        params.train_for_steps = 60
        params.initial_save_rate = 20
        params.batch_size = 32
        params.ppo_epochs = 2
        params.bootstrap_env_steps = 25
        status = train(params, args.env)
        self.assertEqual(status, TrainStatus.SUCCESS)

        root_dir = params.experiment_dir()
        self.assertTrue(os.path.isdir(root_dir))

        enjoy(params, args.env, max_num_episodes=1, max_num_frames=50, fps=1000)
        shutil.rmtree(root_dir)

        self.assertFalse(os.path.isdir(root_dir))

    def test_reachability(self):
        tf.reset_default_graph()

        env = make_doom_env(env_by_name(TEST_ENV_NAME))
        args, params = parse_args_tmax(AgentTMAX.Params)
        reachability_net = ReachabilityNetwork(env, params)

        obs = env.reset()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            probabilities = reachability_net.get_probabilities(sess, [obs], [obs])[0]
            self.assertAlmostEqual(sum(probabilities), 1.0, places=5)  # probs sum up to 1
            reachability = reachability_net.get_reachability(sess, [obs], [obs])[0]
            self.assertEqual(probabilities[1], reachability)

        env.close()

        tf.reset_default_graph()
