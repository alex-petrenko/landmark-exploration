import numpy as np
from unittest import TestCase

from algorithms.tests.test_wrappers import TEST_ENV_NAME
from algorithms.tmax.topological_map import TopologicalMap
from utils.envs.doom.doom_utils import make_doom_env, env_by_name


class TestTMAX(TestCase):
    def test_topological_graph(self):
        env = make_doom_env(env_by_name(TEST_ENV_NAME))
        initial_obs = env.reset()

        m = TopologicalMap(initial_obs)
        self.assertEqual(len(m.landmarks), 1)
        self.assertEqual(len(m.adjacency), 1)
        self.assertEqual(len(m.adjacency[m.curr_landmark_idx]), 0)
        self.assertTrue(np.array_equal(initial_obs, m.curr_landmark))

        obs, _, _, _ = env.step(0)
        new_landmark_idx = m.add_landmark(obs)
        self.assertEqual(new_landmark_idx, 1)

        m.set_curr_landmark(new_landmark_idx)

        self.assertEqual(len(m.landmarks), 2)
        self.assertEqual(len(m.adjacency), 2)
        self.assertEqual(len(m.adjacency[m.curr_landmark_idx]), 1)
        self.assertIn(0, m.adjacency[m.curr_landmark_idx])
        self.assertIn(1, m.adjacency[0])
        self.assertTrue(np.array_equal(obs, m.curr_landmark))

        self.assertEqual(len(m.neighbor_indices()), 2)
        self.assertEqual(len(m.non_neighbor_indices()), 0)
