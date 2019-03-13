import random
from string import ascii_lowercase
from unittest import TestCase

import networkx as nx
import numpy as np

from algorithms.tests.test_wrappers import TEST_ENV_NAME
from algorithms.tmax.topological_map import TopologicalMap
from utils.envs.doom.doom_utils import doom_env_by_name, make_doom_env


class TestGraph(TestCase):
    def test_topological_graph(self):
        env = make_doom_env(doom_env_by_name(TEST_ENV_NAME))
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
        self.assertEqual(len(m.adjacency[m.curr_landmark_idx]), 0)  # directed edge
        self.assertIn(1, m.adjacency[0])
        self.assertTrue(np.array_equal(obs, m.curr_landmark))

        self.assertEqual(len(m.neighbor_indices()), 1)
        self.assertEqual(len(m.non_neighbor_indices()), 1)

    def test_paths(self):
        m = TopologicalMap(np.array(0))

        for i in range(20):
            m.add_landmark(np.array(0))

        m.adjacency[0] = []

        for i, adj in enumerate(m.adjacency):
            for j in range(4):
                rand = random.randint(0, len(m.adjacency) - 1)
                if rand != 0 and rand != i and rand not in adj:
                    adj.append(rand)

        shortest, _ = m.shortest_paths(0)

        relabeling = {}
        for i, s in enumerate(shortest):
            if i != 0 and s != float('inf'):
                new_s = str(s)
                for c in ascii_lowercase:
                    if new_s in relabeling.values():
                        new_s = str(s) + c
                    else:
                        break
                relabeling[i] = new_s
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
