import glob
import math
import os
import pickle as pkl
import random
import shutil
import datetime
from collections import deque
from hashlib import sha1
from os.path import join, isfile

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

import networkx as nx

from algorithms.utils.algo_utils import EPS
from utils.graph import visualize_graph_tensorboard, plot_graph
from utils.timing import Timing
from utils.utils import ensure_contigious, log, ensure_dir_exists, AttrDict


def hash_observation(o):
    """Not the fastest way to do it, but plenty fast enough for our purposes."""
    o = ensure_contigious(o)
    return sha1(o).hexdigest()


def get_position(info):
    pos = None
    if info is not None:
        pos = info.get('pos')
        if pos is not None:
            pos = (pos['agent_x'], pos['agent_y'])
    return pos


def get_angle(info):
    angle = None
    if info is not None:
        pos = info.get('pos')
        if pos is not None:
            angle = pos['agent_a']
    return angle


class TopologicalMap:
    def __init__(self, initial_obs, directed_graph, initial_info=None, verbose=False):
        self._verbose = verbose

        # whether we add edges in both directions or not (directions are always treated separately, hence DiGraph)
        self.directed_graph = directed_graph
        self.graph = nx.DiGraph()

        self.curr_landmark_idx = 0
        self.path_so_far = [0]  # full path traversed during the last or current episode

        # variables needed for online localization
        self.new_landmark_candidate_frames = 0
        self.loop_closure_candidate_frames = 0
        self.closest_landmarks = []

        # number of trajectories that were used to build the map (used in TMAX)
        self.num_trajectories = 0
        # index map from frame index in a trajectory to node index in the resulting map
        self.frame_to_node_idx = dict()

        self.reset(initial_obs, initial_info)

    @staticmethod
    def create_empty():
        return TopologicalMap(np.array(0), directed_graph=False)

    def _add_new_node(self, obs, pos, angle, value_estimate=0.0, num_samples=1, node_id=None):
        if node_id is not None:
            new_landmark_idx = node_id
        else:
            if self.num_landmarks() <= 0:
                new_landmark_idx = 0
            else:
                new_landmark_idx = max(self.graph.nodes) + 1

        assert new_landmark_idx not in self.graph.nodes

        hash_ = hash_observation(obs)
        self.graph.add_node(
            new_landmark_idx,
            obs=obs, hash=hash_, pos=pos, angle=angle,
            value_estimate=value_estimate, num_samples=num_samples, path=(new_landmark_idx,),
        )

        return new_landmark_idx

    def _node_set_path(self, idx):
        self.graph.nodes[idx]['path'] = tuple(self.path_so_far)

    def reset(self, obs, info=None):
        """Create the graph with only one vertex."""
        self.graph.clear()

        self.curr_landmark_idx = self._add_new_node(obs=obs, pos=get_position(info), angle=get_angle(info))
        assert self.curr_landmark_idx == 0
        self.frame_to_node_idx[0] = [0]

        self.new_episode()

    def new_episode(self):
        self.new_landmark_candidate_frames = 0
        self.loop_closure_candidate_frames = 0
        self.closest_landmarks = []
        self.curr_landmark_idx = 0  # assuming we're being put into the exact same starting spot every time
        self.graph.nodes[self.curr_landmark_idx]['added_at'] = 0
        self.path_so_far = [0]

    def relabel_nodes(self):
        """Make sure nodes are labeled from 0 to n-1."""
        self.graph = nx.convert_node_labels_to_integers(self.graph)

    def _log_verbose(self, msg, *args):
        if not self._verbose:
            return
        log.debug(msg, *args)

    @property
    def curr_landmark_obs(self):
        return self.get_observation(self.curr_landmark_idx)

    # noinspection PyUnresolvedReferences
    def get_observation(self, landmark_idx):
        return self.graph.node[landmark_idx]['obs']

    # noinspection PyUnresolvedReferences
    def get_hash(self, landmark_idx):
        return self.graph.node[landmark_idx]['hash']

    # noinspection PyUnresolvedReferences
    def get_info(self, landmark_idx):
        x = y = angle = 0
        try:
            x, y = self.graph.node[landmark_idx]['pos']
            angle = self.graph.node[landmark_idx]['angle']
        except (KeyError, TypeError):
            log.warning(f'No coordinate information in landmark {landmark_idx}')

        pos = {
            'agent_x': x, 'agent_y': y, 'agent_a': angle,
        }
        return {'pos': pos}

    def neighbors(self, landmark_idx):
        return list(nx.neighbors(self.graph, landmark_idx))

    def neighborhood(self):
        neighbors = [self.curr_landmark_idx]
        neighbors.extend(self.neighbors(self.curr_landmark_idx))
        return neighbors

    def reachable_indices(self, start_idx):
        """Run BFS from current landmark to find the list of landmarks reachable from the current landmark."""
        d = [start_idx]
        d.extend(nx.descendants(self.graph, start_idx))
        return d

    def non_neighbors(self, landmark_idx):
        return list(nx.non_neighbors(self.graph, landmark_idx))

    def curr_non_neighbors(self):
        return self.non_neighbors(self.curr_landmark_idx)

    def set_curr_landmark(self, landmark_idx):
        """Replace current landmark with the given landmark. Create necessary edges if needed."""
        if landmark_idx == self.curr_landmark_idx:
            return

        if landmark_idx not in self.neighborhood():
            # create new edges, we found a loop closure!
            self.add_edge(self.curr_landmark_idx, landmark_idx, loop_closure=True)

        self._log_verbose('Change current landmark to %d', landmark_idx)
        self.curr_landmark_idx = landmark_idx
        self.path_so_far.append(landmark_idx)

    def add_landmark(self, obs, info=None, update_curr_landmark=False, action=None):
        new_landmark_idx = self._add_new_node(obs=obs, pos=get_position(info), angle=get_angle(info))
        self.add_edge(self.curr_landmark_idx, new_landmark_idx)
        self._log_verbose('Added new landmark %d', new_landmark_idx)

        if update_curr_landmark:
            prev_landmark_idx = self.curr_landmark_idx
            self.set_curr_landmark(new_landmark_idx)
            self._node_set_path(new_landmark_idx)
            assert self.path_so_far[-1] == new_landmark_idx

            if prev_landmark_idx != self.curr_landmark_idx and action is not None:
                self.graph.adj[prev_landmark_idx][self.curr_landmark_idx]['action'] = action

        return new_landmark_idx

    def add_edge(self, i1, i2, loop_closure=False):
        initial_success = 0.01  # add to params?

        if i2 in self.graph[i1]:
            log.warning('Edge %d-%d already exists (%r)! Overriding!', i1, i2, self.graph[i1])

        self.graph.add_edge(
            i1, i2,
            success=initial_success, last_traversal_frames=math.inf, attempted_traverse=0,
            loop_closure=loop_closure,
        )
        if not self.directed_graph:
            if i1 in self.graph[i2]:
                log.warning('Edge %d-%d already exists (%r)! Overriding!', i2, i1, self.graph[i2])

            self.graph.add_edge(
                i2, i1,
                success=initial_success, last_traversal_frames=math.inf, attempted_traverse=0,
                loop_closure=loop_closure,
            )

    def _remove_edge(self, i1, i2):
        if i2 in self.graph[i1]:
            self.graph.remove_edge(i1, i2)
        if not self.directed_graph:
            if i1 in self.graph[i2]:
                self.graph.remove_edge(i2, i1)

    def remove_edges_from(self, edges):
        for e in edges:
            self._remove_edge(*e)

    def remove_unreachable_vertices(self, from_idx):
        reachable_targets = self.reachable_indices(from_idx)
        remove_vertices = []
        for target_idx in self.graph.nodes():
            if target_idx not in reachable_targets:
                remove_vertices.append(target_idx)

        assert len(remove_vertices) < self.num_landmarks()
        self.graph.remove_nodes_from(remove_vertices)

    def num_edges(self):
        """Helper function for summaries."""
        return self.graph.number_of_edges()

    def num_landmarks(self):
        return self.graph.number_of_nodes()

    def update_edge_traversal(self, i1, i2, success, frames):
        """Update traversal information only for one direction."""
        learning_rate = 0.2

        prev_success = self.graph[i1][i2]['success']
        self.graph[i1][i2]['success'] = (1 - learning_rate) * prev_success + learning_rate * success
        self.graph[i1][i2]['last_traversal_frames'] = frames

    # noinspection PyUnusedLocal
    @staticmethod
    def edge_weight(i1, i2, d, max_probability=1.0):
        success_prob = d['success']
        success_prob = max(EPS, success_prob)
        success_prob = min(max_probability, success_prob)
        return -math.log(success_prob)  # weight of the edge is neg. log probability of traversal success

    def get_path(self, from_idx, to_idx, edge_weight=None):
        if edge_weight is None:
            edge_weight = self.edge_weight

        try:
            return nx.dijkstra_path(self.graph, from_idx, to_idx, weight=edge_weight)
        except nx.exception.NetworkXNoPath:
            return None

    def path_lengths(self, from_idx):
        return nx.shortest_path_length(self.graph, from_idx, weight=self.edge_weight)

    def topological_distances(self, from_idx):
        return nx.shortest_path_length(self.graph, from_idx)

    def topological_neighborhood(self, idx, max_dist):
        """Return set of vertices that are within [0, max_dist] of idx."""
        ego_graph = nx.ego_graph(self.graph, idx, max_dist)
        neighbors = list(ego_graph.nodes)
        return neighbors

    def distances_from(self, another_map):
        """
        Calculate topological distances from all nodes in another map (usually submap) to nodes in this map.
        For all nodes in the intersection of graphs the distance should be 0.
        Solved using BFS (probably there's an algorithm in NX for this).
        """
        q = deque(another_map.graph.nodes)
        distances = {node: 0 for node in another_map.graph.nodes}

        while len(q) > 0:
            node = q.popleft()
            if node not in self.graph:
                continue

            for adj_node in list(self.graph.adj[node]):
                if adj_node in distances:
                    continue

                distances[adj_node] = distances[node] + 1
                q.append(adj_node)

        return distances

    def get_cut_from(self, another_map):
        """
        Return set of edges (cut) that completely separates current map from another_map (usually subgraph).
        """
        distances = self.distances_from(another_map)
        surrounding_vertices = [node for node, d in distances.items() if d == 1]
        cut_edges = []

        for v in surrounding_vertices:
            for adj_v in self.graph.adj[v]:
                assert distances[v] == 1
                if adj_v in another_map.graph:
                    assert distances[adj_v] == 0
                    cut_edges.append((adj_v, v))

        return cut_edges

    @property
    def labeled_graph(self):
        g = self.graph.copy()
        labels = {i: str(i) for i in g.nodes}
        g = nx.relabel_nodes(g, labels)
        return g

    def save_checkpoint(
            self, checkpoint_dir, map_img=None, coord_limits=None, num_to_keep=2, is_sparse=False, verbose=False,
    ):
        """Verbose mode also dumps all the landmark observations and the graph structure into the directory."""
        t = Timing()
        with t.timeit('map_checkpoint'):
            results = AttrDict()

            prefix = '.map_'
            timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            dir_name = f'{prefix}{timestamp}'
            map_dir = join(checkpoint_dir, dir_name)

            if os.path.isdir(map_dir):
                log.warning('Warning: map checkpoint %s already exists! Overwriting...')
                shutil.rmtree(map_dir)

            map_dir = ensure_dir_exists(map_dir)

            with open(join(map_dir, 'topo_map.pkl'), 'wb') as fobj:
                pkl.dump(self.__dict__, fobj, 2)

            if verbose:
                map_extra = ensure_dir_exists(join(map_dir, '.map_verbose'))
                for node in self.graph.nodes:
                    obs = self.get_observation(node)
                    obs_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
                    obs_bgr_bigger = cv2.resize(obs_bgr, (420, 420), interpolation=cv2.INTER_NEAREST)
                    cv2.imwrite(join(map_extra, f'{node:03d}.jpg'), obs_bgr_bigger)

                figure = plot_graph(
                    self.graph,
                    layout='pos', map_img=map_img, limits=coord_limits, topological_map=True, is_sparse=is_sparse,
                )
                graph_filename = join(map_extra, 'graph.png')
                with open(graph_filename, 'wb') as graph_fobj:
                    plt.savefig(graph_fobj, format='png')
                figure.clear()

                results.graph_filename = graph_filename

            assert num_to_keep > 0
            previous_checkpoints = glob.glob(f'{checkpoint_dir}/{prefix}*')
            previous_checkpoints.sort()
            previous_checkpoints = deque(previous_checkpoints)

            while len(previous_checkpoints) > num_to_keep:
                checkpoint_to_delete = previous_checkpoints[0]
                log.info('Deleting old map checkpoint %s', checkpoint_to_delete)
                shutil.rmtree(checkpoint_to_delete)
                previous_checkpoints.popleft()

        log.info('Map save checkpoint took %s', t)
        return results

    def maybe_load_checkpoint(self, checkpoint_dir):
        prefix = '.map_'
        all_map_checkpoints = glob.glob(f'{checkpoint_dir}/{prefix}*')

        if len(all_map_checkpoints) <= 0:
            log.debug('No map checkpoints found, starting from empty map')
            return

        all_map_checkpoints.sort()
        latest_checkpoint = all_map_checkpoints[-1]

        fname = 'topo_map.pkl'
        full_path = join(latest_checkpoint, fname)
        if not isfile(full_path):
            return

        log.debug('Load env map from file %s', full_path)
        with open(full_path, 'rb') as fobj:
            topo_map_dict = pkl.load(fobj)
            self.load_dict(topo_map_dict)

    def load_dict(self, topo_map_dict):
        self.__dict__.update(topo_map_dict)


def map_summaries(maps, env_steps, summary_writer, section, map_img=None, coord_limits=None, is_sparse=False):
    if None in maps:
        return

    # summaries related to episodic memory (maps)
    num_landmarks = [m.num_landmarks() for m in maps]
    num_edges = [m.num_edges() for m in maps]

    num_neighbors = []
    for m in maps:
        node = random.choice(list(m.graph.nodes))
        num_neighbors.append(len(m.neighbors(node)))

    avg_num_landmarks = sum(num_landmarks) / len(num_landmarks)
    avg_num_neighbors = sum(num_neighbors) / len(num_neighbors)
    avg_num_edges = sum(num_edges) / len(num_edges)

    summary = tf.Summary()

    def curiosity_summary(tag, value):
        summary.value.add(tag=f'{section}/{tag}', simple_value=float(value))

    curiosity_summary('avg_landmarks', avg_num_landmarks)
    curiosity_summary('max_landmarks', max(num_landmarks))
    curiosity_summary('avg_neighbors', avg_num_neighbors)
    curiosity_summary('max_neighbors', max(num_neighbors))
    curiosity_summary('avg_edges', avg_num_edges)
    curiosity_summary('max_edges', max(num_edges))

    summary_writer.add_summary(summary, env_steps)

    num_maps_to_plot = min(2, len(maps))
    maps_for_summary = random.sample(maps, num_maps_to_plot)

    max_graph_idx = 0
    for i, m in enumerate(maps):
        if m.num_landmarks() > maps[max_graph_idx].num_landmarks():
            max_graph_idx = i

    max_graph_summary = visualize_graph_tensorboard(
        maps[max_graph_idx].labeled_graph,
        tag=f'{section}/max_graph', map_img=map_img, coord_limits=coord_limits, is_sparse=is_sparse,
    )
    summary_writer.add_summary(max_graph_summary, env_steps)

    if len(maps) > 1:
        for i, map_for_summary in enumerate(maps_for_summary):
            random_graph_summary = visualize_graph_tensorboard(
                map_for_summary.labeled_graph,
                tag=f'{section}/random_graph_{i}',
                map_img=map_img, coord_limits=coord_limits, is_sparse=is_sparse,
            )
            summary_writer.add_summary(random_graph_summary, env_steps)
