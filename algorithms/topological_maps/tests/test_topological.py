import copy
import math
import random
from string import ascii_lowercase
from unittest import TestCase

import numpy as np
import networkx as nx

from algorithms.tests.test_wrappers import TEST_ENV_NAME
from algorithms.topological_maps.topological_map import TopologicalMap, hash_observation
from utils.envs.doom.doom_utils import doom_env_by_name, make_doom_env
from utils.graph import plot_graph
from utils.timing import Timing
from utils.utils import log


class TestGraph(TestCase):
    def test_topological_graph(self):
        env = make_doom_env(doom_env_by_name(TEST_ENV_NAME))
        initial_obs = env.reset()

        m = TopologicalMap(initial_obs, True)
        self.assertEqual(m.num_landmarks(), 1)
        self.assertEqual(len(m.neighborhood()), 1)
        self.assertTrue(np.array_equal(initial_obs, m.curr_landmark_obs))

        obs, _, _, _ = env.step(0)
        new_landmark_idx = m.add_landmark(obs)
        self.assertEqual(new_landmark_idx, 1)

        m.set_curr_landmark(new_landmark_idx)

        self.assertEqual(m.num_landmarks(), 2)
        self.assertEqual(m.num_edges(), 1)
        self.assertTrue(m.graph.has_edge(0, 1))
        self.assertTrue(np.array_equal(obs, m.curr_landmark_obs))

        self.assertEqual(len(m.neighborhood()), 1)
        self.assertEqual(len(m.curr_non_neighbors()), 1)

        self.assertEqual(sorted(m.reachable_indices(1)), [1])
        self.assertEqual(sorted(m.reachable_indices(0)), [0, 1])

        path = m.get_path(0, 1)
        self.assertEqual(path, [0, 1])

    def test_hash_performance(self):
        x = np.empty((84, 84, 3), dtype=np.uint8)
        t = Timing()
        with t.timeit('hashes'):
            for i in range(1000):
                hash_observation(x)
        log.debug('Hash 1k images %s', t)

    def test_shortest(self):
        m = TopologicalMap(np.array(0), directed_graph=False)
        for i in range(5):
            m.add_landmark(np.array(0))

        m.graph.remove_edges_from(list(m.graph.edges))

        m._add_edge(0, 1)
        m._add_edge(0, 2)
        m._add_edge(1, 3)
        m._add_edge(2, 3)

        path_lengths = m.path_lengths(0)
        log.debug('Path lenghts from %d: %r', 0, path_lengths)

        reachable_indices_0 = m.reachable_indices(0)
        log.debug('Reachable from %d: %r', 0, reachable_indices_0)

        reachable_indices_1 = m.reachable_indices(1)
        log.debug('Reachable from %d: %r', 1, reachable_indices_1)

        self.assertEqual(sorted(reachable_indices_0), sorted(reachable_indices_1))
        self.assertEqual(sorted(reachable_indices_0), [0, 1, 2, 3])

        topological_distances = m.topological_distances(0)
        log.debug('Topological distances: %r', topological_distances)

        path = m.get_path(0, 3)
        log.debug('Path from 0 to 3 is %r', path)

        m.update_edge_traversal(0, 1, 0, 10)

        path = m.get_path(0, 3)
        log.debug('Path from 0 to 3 is %r', path)
        self.assertEqual(path, [0, 2, 3])

        m.update_edge_traversal(2, 3, 0, 10)
        m.update_edge_traversal(2, 3, 0, 10)

        path = m.get_path(0, 3)
        log.debug('Path from 0 to 3 is %r', path)
        self.assertEqual(path, [0, 1, 3])

        m.update_edge_traversal(2, 3, 1, 10)
        m.update_edge_traversal(2, 3, 1, 10)
        m.update_edge_traversal(0, 2, 1, 10)

        path = m.get_path(0, 3)
        log.debug('Path from 0 to 3 is %r', path)
        self.assertEqual(path, [0, 2, 3])

        m._remove_edge(2, 3)

        path = m.get_path(0, 3)
        log.debug('Path from 0 to 3 is %r', path)
        self.assertEqual(path, [0, 1, 3])

        for i in range(5):
            m.update_edge_traversal(0, 1, 0, 10)

    def test_paths(self):
        m = TopologicalMap(np.array(0), directed_graph=True)

        for i in range(20):
            m.add_landmark(np.array(0))

        for v in m.neighbors(0):
            m.graph.remove_edge(0, v)

        for i in range(m.num_landmarks()):
            if i > m.num_landmarks() - 3:
                continue

            if random.random() < 0.9:
                for j in range(random.randint(1, 3)):
                    rand = random.randint(0, m.num_landmarks())
                    if rand != 0 and rand != i:
                        m._add_edge(i, rand)

        shortest = nx.shortest_path(m.graph, 0)
        shortest = [len(path) for path in shortest.values()]
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
                    new_s = f'{s:.2f}' + c
                    if new_s not in relabeling.values():
                        relabeling[i] = new_s
                        break
            else:
                relabeling[i] = s

        graph = m.labeled_graph
        new_graph = nx.relabel_nodes(graph, relabeling)

        figure = plot_graph(new_graph, layout='kamada_kawai', node_size=400)

        show = False
        if show:
            from matplotlib import pyplot as plt
            plt.show()

        figure.clear()

    def test_plot_coordinates(self):
        m = TopologicalMap(
            np.array(0), directed_graph=True, initial_info={'pos': {'agent_x': 300, 'agent_y': 400, 'agent_a': 0}},
        )

        for i in range(1, 4):
            for j in range(1, 4):
                m.add_landmark(np.array(0), {'pos': {'agent_x': 300 + 50*i, 'agent_y': 400 + 100*j, 'agent_a': 10}})

        graph = m.labeled_graph
        figure = plot_graph(graph, layout='pos')

        show = False
        if show:
            from matplotlib import pyplot as plt
            plt.show()

        figure.clear()

    def test_map_operations(self):
        obs = np.array(0)
        m = TopologicalMap(obs, directed_graph=False)

        # constructing test "persistent map"
        m.add_landmark(obs)  # 0 <-> 1
        m.add_landmark(obs)  # 0 <-> 2

        m.set_curr_landmark(1)
        m.add_landmark(obs)  # 1 <-> 3
        m.set_curr_landmark(2)  # 1 <-> 2
        m.set_curr_landmark(3)  # 2 <-> 3

        undirected_edges = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)]
        for e in undirected_edges:
            self.assertIn(e, m.graph.edges)
            self.assertIn(reversed(e), m.graph.edges)
        self.assertEqual(m.num_edges(), 2 * len(undirected_edges))

        new_map = copy.deepcopy(m)

        # constructing test "new map" (after exploration we have some new edges)
        new_map.new_episode()
        new_map.set_curr_landmark(1)
        new_map.add_landmark(obs)  # 1 <-> 4

        new_map.set_curr_landmark(4)
        new_map.add_landmark(obs)  # 4 <-> 5

        new_map.set_curr_landmark(5)
        new_map.add_landmark(obs)  # 5 <-> 6

        new_map.set_curr_landmark(4)
        new_map.set_curr_landmark(3)  # 3 <-> 4

        new_map.add_landmark(obs)  # 3 <-> 7

        new_map.set_curr_landmark(7)
        new_map.add_landmark(obs)  # 7 <-> 8

        new_map.set_curr_landmark(7)
        new_map.add_landmark(obs)  # 7 <-> 9

        new_map.set_curr_landmark(3)
        new_map.set_curr_landmark(2)
        new_map.set_curr_landmark(0)
        new_map.add_landmark(obs)  # 0 <-> 10

        undirected_edges.extend([
            (0, 10), (1, 4), (3, 4), (3, 7), (4, 5), (5, 6), (7, 8), (7, 9),
        ])
        for e in undirected_edges:
            self.assertIn(e, new_map.graph.edges)
            self.assertIn(reversed(e), new_map.graph.edges)
        self.assertEqual(new_map.num_edges(), 2 * len(undirected_edges))

        distances = new_map.distances_from(m)
        self.assertTrue(all(distances[n] == 0 for n in m.graph.nodes))
        self.assertTrue(all(distances[n] == 1 for n in [4, 7, 10]))
        self.assertTrue(all(distances[n] == 2 for n in [5, 8, 9]))
        self.assertTrue(all(distances[n] == 3 for n in [6]))

        candidate_edges = new_map.get_cut_from(m)
        self.assertEqual(sorted(candidate_edges), [(0, 10), (1, 4), (3, 4), (3, 7)])
        self.assertEqual(len(candidate_edges), 4)

        candidate_edges = m.get_cut_from(new_map)
        self.assertEqual(len(candidate_edges), 0)
