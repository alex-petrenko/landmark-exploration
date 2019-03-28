import math
import random
from hashlib import sha1

import tensorflow as tf

import networkx as nx

from algorithms.algo_utils import EPS
from utils.graph import visualize_graph_tensorboard
from utils.utils import ensure_contigious, log


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

        # variables needed for online localization
        self.new_landmark_candidate_frames = 0
        self.closest_landmarks = []

        self.reset(initial_obs, initial_info)

    def _add_node(self, idx, obs, hash_, pos, angle, locomotion_goal, value_estimate, num_samples):
        self.graph.add_node(
            idx,
            obs=obs, hash=hash_, pos=pos, angle=angle, locomotion_goal=locomotion_goal,
            value_estimate=value_estimate, num_samples=num_samples,
        )

    def reset(self, obs, info=None):
        """Create the graph with only one vertex."""
        self.graph.clear()

        self._add_node(
            0,
            obs=obs, hash_=hash_observation(obs), pos=get_position(info), angle=get_angle(info),
            locomotion_goal=0,
            value_estimate=0.0, num_samples=1,
        )

        self.curr_landmark_idx = 0

        self.new_episode()

    def new_episode(self):
        self.new_landmark_candidate_frames = 0
        self.closest_landmarks = []
        self.curr_landmark_idx = 0  # assuming we're being put into the exact same spot every time

        self.relabel_nodes()  # make sure nodes are labeled from 0 to n-1

    def relabel_nodes(self):
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

    def curr_non_neighbors(self):
        return list(nx.non_neighbors(self.graph, self.curr_landmark_idx))

    def set_curr_landmark(self, landmark_idx):
        """Replace current landmark with the given landmark. Create necessary edges if needed."""

        if landmark_idx == self.curr_landmark_idx:
            return

        if landmark_idx not in self.neighborhood():
            # create new edges, we found a loop closure!
            self._add_edge(self.curr_landmark_idx, landmark_idx)

        self._log_verbose('Change current landmark to %d', landmark_idx)
        self.curr_landmark_idx = landmark_idx

    def add_landmark(self, obs, info=None):
        new_landmark_idx = max(self.graph.nodes) + 1
        assert new_landmark_idx not in self.graph.nodes

        self._add_node(
            new_landmark_idx,
            obs=obs, hash_=hash_observation(obs), pos=get_position(info), angle=get_angle(info),
            locomotion_goal=0,
            value_estimate=0.0, num_samples=1,
        )

        self._add_edge(self.curr_landmark_idx, new_landmark_idx)
        self._log_verbose('Added new landmark %d', new_landmark_idx)
        return new_landmark_idx

    def _add_edge(self, i1, i2):
        initial_success = 0.5  # add to params?

        self.graph.add_edge(i1, i2, success=initial_success, last_traversal_frames=math.inf, traversed=False)
        if not self.directed_graph:
            self.graph.add_edge(i2, i1, success=initial_success, last_traversal_frames=math.inf, traversed=False)

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
        prev_value = self.graph[i1][i2]['success']
        self.graph[i1][i2]['success'] = 0.5 * (prev_value + success)
        self.graph[i1][i2]['last_traversal_frames'] = frames
        self.graph[i1][i2]['traversed'] = True

    # noinspection PyUnusedLocal
    @staticmethod
    def _edge_weight(i1, i2, d):
        success_prob = d['success']
        success_prob = max(EPS, success_prob)
        return -math.log(success_prob)  # weight of the edge is neg. log probability of traversal success

    def get_path(self, from_idx, to_idx):
        try:
            return nx.dijkstra_path(self.graph, from_idx, to_idx, weight=self._edge_weight)
        except nx.exception.NetworkXNoPath:
            return None

    def path_lengths(self, from_idx):
        return nx.shortest_path_length(self.graph, from_idx, weight=self._edge_weight)

    def topological_distances(self, from_idx):
        return nx.shortest_path_length(self.graph, from_idx)

    @property
    def labeled_graph(self):
        g = self.graph.copy()
        labels = {i: str(i) for i in g.nodes}
        g = nx.relabel_nodes(g, labels)
        return g


def map_summaries(maps, env_steps, summary_writer, section):
    if None in maps:
        return

    # summaries related to episodic memory (maps)
    num_landmarks = [m.num_landmarks() for m in maps]
    num_neighbors = [len(m.neighborhood()) for m in maps]
    num_edges = [m.num_edges() for m in maps]

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

    num_maps_to_plot = min(3, len(maps))
    maps_for_summary = random.sample(maps, num_maps_to_plot)

    for i, map_for_summary in enumerate(maps_for_summary):
        random_graph_summary = visualize_graph_tensorboard(
            map_for_summary.labeled_graph, tag=f'{section}/random_graph_{i}',
        )
        summary_writer.add_summary(random_graph_summary, env_steps)

    max_graph_idx = 0
    for i, m in enumerate(maps):
        if m.num_landmarks() > maps[max_graph_idx].num_landmarks():
            max_graph_idx = i

    max_graph_summary = visualize_graph_tensorboard(maps[max_graph_idx].labeled_graph, tag=f'{section}/max_graph')
    summary_writer.add_summary(max_graph_summary, env_steps)
