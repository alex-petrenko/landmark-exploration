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
from algorithms.tmax.tmax_utils import parse_args_tmax
from algorithms.tmax.train_tmax import train
from algorithms.utils.trajectory import TrajectoryBuffer
from utils.envs.doom.doom_utils import make_doom_env, doom_env_by_name
from utils.utils import experiments_dir, ensure_dir_exists


class TestTMAX(TestCase):
    def tmax_train_run(self, env_name=None):
        test_dir_name = self.__class__.__name__

        args, params = parse_args_tmax(AgentTMAX.Params)
        if env_name is not None:
            args.env = env_name

        params.experiments_root = test_dir_name
        params.num_envs = 16
        params.train_for_steps = 60
        params.initial_save_rate = 20
        params.batch_size = 32
        params.ppo_epochs = 2
        params.distance_bootstrap = 10
        params.stage_duration = 100

        tmax_train_dir = join(experiments_dir(), params.experiments_root)
        ensure_dir_exists(tmax_train_dir)
        shutil.rmtree(tmax_train_dir)

        status = train(params, args.env)
        self.assertEqual(status, TrainStatus.SUCCESS)

        root_dir = params.experiment_dir()
        self.assertTrue(os.path.isdir(root_dir))

        enjoy(params, args.env, max_num_episodes=1, max_num_frames=50)
        shutil.rmtree(tmax_train_dir)

        self.assertFalse(os.path.isdir(root_dir))

    def test_tmax_train_run(self):
        self.tmax_train_run()

    def test_tmax_train_run_goal(self):
        self.tmax_train_run(env_name='doom_maze_goal')

    def test_locomotion(self):
        g = tf.Graph()

        env = make_doom_env(doom_env_by_name(TEST_ENV_NAME))
        args, params = parse_args_tmax(AgentTMAX.Params)

        with g.as_default():
            locomotion_net = LocomotionNetwork(env, params)

        obs = env.reset()

        with tf.Session(graph=g) as sess:
            sess.run(tf.global_variables_initializer())

            action = locomotion_net.navigate(sess, [obs], [obs], [obs])[0]
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, env.action_space.n)

        env.close()

        g.finalize()

    def test_trajectory(self):
        num_envs = 10
        buffer = TrajectoryBuffer(num_envs)
        self.assertEqual(len(buffer.complete_trajectories), 0)
