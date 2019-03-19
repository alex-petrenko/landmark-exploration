import math
import random
from string import ascii_lowercase
from unittest import TestCase

import networkx as nx
import numpy as np

from algorithms.tests.test_wrappers import TEST_ENV_NAME
from algorithms.tmax.topological_map import TopologicalMap
from utils.envs.doom.doom_utils import doom_env_by_name, make_doom_env
from utils.utils import log


class TestGraph(TestCase):
    def test_topological_graph(self):
        env = make_doom_env(doom_env_by_name(TEST_ENV_NAME))
        initial_obs = env.reset()

        m = TopologicalMap(initial_obs, directed_graph=True)
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
        self.assertEqual(len(m.adjacency[m.curr_landmark_idx]), 0)  # directed edge
        self.assertIn(1, m.adjacency[0])
        self.assertTrue(np.array_equal(obs, m.curr_landmark))

        self.assertEqual(len(m.neighbor_indices()), 1)
        self.assertEqual(len(m.non_neighbor_indices()), 1)

        self.assertEqual(sorted(m.reachable_indices(1)), [1])
        self.assertEqual(sorted(m.reachable_indices(0)), [0, 1])

        path = m.get_path(0, 1)
        self.assertEqual(path, [0, 1])

    def test_paths(self):
        m = TopologicalMap(np.array(0), directed_graph=True)

        for i in range(20):
            m.add_landmark(np.array(0))

        m.adjacency[0] = []

        for i, adj in enumerate(m.adjacency):
            if i > len(m.adjacency) - 3:
                continue

            if random.random() < 0.9:
                for j in range(random.randint(1, 3)):
                    rand = random.randint(0, len(m.adjacency) - 1)
                    if rand != 0 and rand != i and rand not in adj:
                        adj.append(rand)

        shortest, _ = m.shortest_paths(0)
        reachable = m.reachable_indices(0)
        self.assertGreaterEqual(len(reachable), 1)
        log.debug('Reachable vertices: %r', reachable)

        max_path_idx = -1
        max_path_length = -1
        for i, path_length in enumerate(shortest):
            if path_length == math.inf:
                continue
            if path_length > max_path_length:
                max_path_length = path_length
                max_path_idx = i

        path = m.get_path(0, max_path_idx)
        log.debug('Shortest path from %d to %d is %r', 0, max_path_idx, path)

        relabeling = {}
        for i, s in enumerate(shortest):
            if i != 0:
                for c in ascii_lowercase:
                    new_s = str(s) + c
                    if new_s not in relabeling.values():
                        relabeling[i] = new_s
                        break
            else:
                relabeling[i] = s

        graph = m.to_nx_graph()
        new_graph = nx.relabel_nodes(graph, relabeling)

        from matplotlib import pyplot as plt
        figure = plt.gcf()
        figure.clear()

        nx.draw(
            new_graph, nx.kamada_kawai_layout(new_graph),
            node_size=100, node_color=list(graph.nodes), edge_color='#cccccc', cmap=plt.cm.get_cmap('plasma'),
            with_labels=True, font_color='#ffffff', font_size=7,
        )
        plt.show()
        figure = plt.gcf()
        figure.clear()
