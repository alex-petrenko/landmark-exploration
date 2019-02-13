import numpy as np
import os
import shutil
from os.path import join
from unittest import TestCase

import tensorflow as tf

from algorithms.agent import TrainStatus
from algorithms.tests.test_wrappers import TEST_ENV_NAME
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.enjoy_tmax import enjoy
from algorithms.tmax.locomotion import LocomotionNetwork
from algorithms.tmax.reachability import ReachabilityNetwork
from algorithms.tmax.tmax_utils import parse_args_tmax
from algorithms.tmax.topological_map import TopologicalMap
from algorithms.tmax.train_tmax import train
from algorithms.tmax.trajectory import TrajectoryBuffer
from utils.envs.doom.doom_utils import make_doom_env, doom_env_by_name
from utils.utils import experiments_dir


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
        shutil.rmtree(join(experiments_dir(), params.experiments_root))

        self.assertFalse(os.path.isdir(root_dir))

    def test_reachability(self):
        g = tf.Graph()

        env = make_doom_env(doom_env_by_name(TEST_ENV_NAME))
        args, params = parse_args_tmax(AgentTMAX.Params)

        with g.as_default():
            reachability_net = ReachabilityNetwork(env, params)

        obs = env.reset()

        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())
            probabilities = reachability_net.get_probabilities(sess, [obs], [obs])[0]
            self.assertAlmostEqual(sum(probabilities), 1.0, places=5)  # probs sum up to 1
            reachability = reachability_net.get_reachability(sess, [obs], [obs])[0]
            self.assertEqual(probabilities[1], reachability)

        env.close()

        g.finalize()

    def test_locomotion(self):
        g = tf.Graph()

        env = make_doom_env(doom_env_by_name(TEST_ENV_NAME))
        args, params = parse_args_tmax(AgentTMAX.Params)

        with g.as_default():
            locomotion_net = LocomotionNetwork(env, params)

        obs = env.reset()

        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())

            action = locomotion_net.navigate(sess, [obs], [obs])[0]
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, env.action_space.n)

        env.close()

        g.finalize()

    def test_trajectory(self):
        num_envs = 10
        maps = [TopologicalMap(np.zeros([1])) for _ in range(num_envs)]
        buffer = TrajectoryBuffer(num_envs)
        self.assertEqual(len(buffer.current_trajectories[0].landmarks), 0)

        buffer.add([0] * num_envs, [0] * num_envs, [False] * 10, maps, [True] * 10)
        buffer.add([0] * num_envs, [0] * num_envs, [False] * 10, maps, [False] * 9 + [True])
        buffer.add([0] * num_envs, [0] * num_envs, [True] * 10, maps, [False] * 9)

        self.assertEqual(len(buffer.complete_trajectories), num_envs)

        for i in range(num_envs):
            if len(buffer.complete_trajectories[i].landmarks) > 1:
                self.assertEqual(i, 9)
                self.assertEqual(buffer.complete_trajectories[i].landmarks, [0, 1])
