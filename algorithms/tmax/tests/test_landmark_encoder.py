import time
from unittest import TestCase

import numpy as np

from algorithms.agent import AgentRandom
from algorithms.tmax.topological_map import hash_observation
from utils.envs.atari.atari_utils import make_atari_env, atari_env_by_name
from utils.utils import log


class TestLandmarkEncoder(TestCase):
    @staticmethod
    def make_env():
        return make_atari_env(atari_env_by_name('atari_montezuma'))

    def test_hashing(self):
        env = self.make_env()
        env.reset()
        agent = AgentRandom(self.make_env, {})

        trajectory = []
        n_obs = 200
        for i in range(n_obs):
            obs, _, _, _ = env.step(agent.best_action())
            trajectory.append(obs)

        start_hashing = time.time()
        hashes = []
        for obs in trajectory:
            obs_hash = hash_observation(obs)
            hashes.append(obs_hash)
        log.debug('Took %.3f seconds to hash %d observations', time.time() - start_hashing, n_obs)

        self.assertEqual(len(trajectory), len(hashes))

        for i in range(n_obs):
            for j in range(n_obs):
                if np.array_equal(trajectory[i], trajectory[j]):
                    self.assertEqual(hashes[i], hashes[j])
                else:
                    self.assertNotEqual(hashes[i], hashes[j])
