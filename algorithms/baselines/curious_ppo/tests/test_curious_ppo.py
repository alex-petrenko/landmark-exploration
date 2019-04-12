import os
import shutil
from os.path import join
from unittest import TestCase

from algorithms.agent import TrainStatus
from algorithms.baselines.curious_ppo.agent_curious_ppo import AgentCuriousPPO
from algorithms.baselines.curious_ppo.curious_ppo_utils import parse_args_curious_ppo
from algorithms.baselines.curious_ppo.enjoy_curious_ppo import enjoy
from algorithms.baselines.curious_ppo.train_curious_ppo import train
from utils.utils import experiments_dir


class TestPPO(TestCase):
    def curious_ppo_train_run(self, env_name=None):
        test_dir_name = self.__class__.__name__

        args, params = parse_args_curious_ppo(AgentCuriousPPO.Params)
        if env_name is not None:
            args.env = env_name
        params.experiments_root = test_dir_name
        params.num_envs = 16
        params.train_for_steps = 60
        params.initial_save_rate = 20
        params.batch_size = 32
        params.ppo_epochs = 2
        params.curiosity_type = 'ecr_map'
        params.reachability_bootstrap = 10
        params.reachability_train_interval = 5
        params.use_env_map = False
        status = train(params, args.env)
        self.assertEqual(status, TrainStatus.SUCCESS)

        root_dir = params.experiment_dir()
        self.assertTrue(os.path.isdir(root_dir))

        enjoy(params, args.env, max_num_episodes=1, max_num_frames=50, fps=1000)
        shutil.rmtree(join(experiments_dir(), params.experiments_root))

        self.assertFalse(os.path.isdir(root_dir))

    def test_curious_ppo_train_run(self):
        self.curious_ppo_train_run()

    def test_curious_ppo_train_run_goal(self):
        self.curious_ppo_train_run(env_name='doom_maze_goal')
