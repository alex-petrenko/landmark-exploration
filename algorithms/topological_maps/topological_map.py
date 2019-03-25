import math
import random
from hashlib import sha1

import tensorflow as tf

import networkx as nx

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

        if directed_graph:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()

        self.positions = None
        self.curr_landmark_idx = 0

        # variables needed for online localization
        self.new_landmark_candidate_frames = 0
        self.closest_landmarks = []

        self.reset(initial_obs, initial_info)

    def reset(self, obs, info=None):
        """Create the graph with only one vertex."""
        self.graph.clear()

        self.graph.add_node(0, obs=obs, hash=hash_observation(obs), pos=get_position(info), angle=get_angle(info))
        self.curr_landmark_idx = 0

        self.new_episode()

    def new_episode(self):
        self.new_landmark_candidate_frames = 0
        self.closest_landmarks = []
        self.curr_landmark_idx = 0  # assuming we're being put into the exact same spot every time

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

        self.graph.add_node(
            new_landmark_idx, obs=obs, hash=hash_observation(obs), pos=get_position(info), angle=get_angle(info),
        )

        self._add_edge(self.curr_landmark_idx, new_landmark_idx)
        self._log_verbose('Added new landmark %d', new_landmark_idx)
        return new_landmark_idx

    def _add_edge(self, i1, i2):
        self.graph.add_edge(i1, i2, success=0.5)

    def _remove_edge(self, i1, i2):
        self.graph.remove_edge(i1, i2)

    def num_edges(self):
        """Helper function for summaries."""
        return self.graph.number_of_edges()

    def num_landmarks(self):
        return self.graph.number_of_nodes()

    def update_edge_traversal(self, i1, i2, success):
        prev_value = self.graph[i1][i2]['success']
        self.graph[i1][i2]['success'] = 0.5 * (prev_value + success)

    def _prune_edges(self, threshold=0.2):
        """Remove edges with very low chance of traversal success"""
        remove_list = []
        for i1, i2, data in self.graph.edges(data=True):
            if data['success'] <= threshold:
                remove_list.append((i1, i2))

        if len(remove_list) > 0:
            log.info('Prune: removing edges %r', remove_list)
            self.graph.remove_edges_from(remove_list)

    def _prune_nodes(self, chance=0.1):
        num_to_remove = int(self.num_landmarks() * chance)
        remove_list = random.sample(self.graph.nodes, num_to_remove)
        if 0 in remove_list:
            remove_list.remove(0)
        self.graph.remove_nodes_from(remove_list)

    def get_path(self, from_idx, to_idx):
        # noinspection PyUnusedLocal
        def edge_weight(i1, i2, d):
            return -math.log(d['success'])

        try:
            return nx.dijkstra_path(self.graph, from_idx, to_idx, weight=edge_weight)
        except nx.exception.NetworkXNoPath:
            return None

    def topological_distances(self, from_idx):
        return nx.shortest_path_length(self.graph, from_idx)

    @property
    def labeled_graph(self):
        g = self.graph.copy()
        labels = {i: str(i) for i in g.nodes}
        g = nx.relabel_nodes(g, labels)
        return g


def map_summaries(maps, env_steps, summary_writer, section):
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

    num_maps_to_plot = 3
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
