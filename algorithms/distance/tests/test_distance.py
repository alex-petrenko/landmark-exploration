import random
import shutil
from unittest import TestCase

import numpy as np

from algorithms.tests.test_wrappers import TEST_ENV_NAME
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
from algorithms.utils.buffer import Buffer
from utils.envs.doom.doom_utils import doom_env_by_name, make_doom_env
from utils.timing import Timing
from utils.utils import log


class TestDistance(TestCase):
    def test_dist_training(self):
        t = Timing()

        def make_env():
            return make_doom_env(doom_env_by_name(TEST_ENV_NAME))

        params = AgentTMAX.Params('__test_dist_train__')
        params.distance_target_buffer_size = 2000

        with t.timeit('generate_data'):
            # first: generate fake random data
            buffer = Buffer()

            obs1 = np.full([84, 84, 3], 0, dtype=np.uint8)
            obs1[:, :, 1] = 255
            obs2 = np.full([84, 84, 3], 0, dtype=np.uint8)
            obs2[:, :, 2] = 255

            data_size = params.distance_target_buffer_size
            for i in range(data_size):
                same = i % 2 == 0
                if same:
                    if random.random() < 0.5:
                        obs_first = obs_second = obs1
                    else:
                        obs_first = obs_second = obs2
                else:
                    obs_first, obs_second = obs1, obs2
                    if random.random() < 0.5:
                        obs_second, obs_first = obs_first, obs_second

                buffer.add(obs_first=obs_first, obs_second=obs_second, labels=0 if same else 1)

        with t.timeit('init'):
            agent = AgentTMAX(make_env, params)
            agent.initialize()

            params.distance_train_epochs = 1
            params.distance_batch_size = 512
            agent.distance.train(buffer, 1, agent)

        with t.timeit('train'):
            params.distance_train_epochs = 2
            params.distance_batch_size = 128
            agent.distance.train(buffer, 1, agent, t)

        agent.finalize()

        log.info('Timing: %s', t)
        shutil.rmtree(params.experiment_dir())
