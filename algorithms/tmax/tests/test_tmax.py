import os
import shutil
from unittest import TestCase

from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.enjoy_tmax import enjoy
from algorithms.tmax.tmax_utils import parse_args_tmax
from algorithms.tmax.train_tmax import train


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
        train(params, args.env)

        root_dir = params.experiment_dir()
        self.assertTrue(os.path.isdir(root_dir))

        enjoy(params, args.env, max_num_episodes=1, fps=2000)
        shutil.rmtree(root_dir)

        self.assertFalse(os.path.isdir(root_dir))
