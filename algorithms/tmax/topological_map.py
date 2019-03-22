from hashlib import sha1
import math
import random

import networkx as nx

from utils.utils import ensure_contigious, log


def hash_observation(o):
    """Not the fastest way to do it, but plenty fast enough for our purposes."""
    o = ensure_contigious(o)
    return sha1(o).hexdigest()


def get_position(info):
    pos = None
    if info is not None:
        pos = info.get('pos')
        pos = (pos['agent_x'], pos['agent_y'])
    return pos


def get_angle(info):
    angle = None
    if info is not None:
        angle = info.get('pos')['agent_a']
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

    def new_episode(self, prune_edge_threshold=0.2, prune_vertex_chance=0.05):
        self.new_landmark_candidate_frames = 0
        self.closest_landmarks = []
        self.curr_landmark_idx = 0
        self._prune_edges(threshold=prune_edge_threshold)
        self._prune_nodes(chance=prune_vertex_chance)

    def _log_verbose(self, msg, *args):
        if not self._verbose:
            return
        log.debug(msg, *args)

    @property
    def curr_landmark_obs(self):
        return self.get_observation(self.curr_landmark_idx)

    def get_observation(self, landmark_idx):
        return nx.get_node_attributes(self.graph, 'obs')[landmark_idx]

    def get_hashes(self, landmark_idx):
        return nx.get_node_attributes(self.graph, 'hash')[landmark_idx]

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
        new_landmark_idx = self.graph.number_of_nodes()
        self.graph.add_node(new_landmark_idx, obs=obs, hash=hash_observation(obs), pos=get_position(info), angle=get_angle(info))

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
        def edge_weight(i1, i2, d):
            return -math.log(d['success'])

        try:
            return nx.dijkstra_path(self.graph, from_idx, to_idx, weight=edge_weight)
        except nx.exception.NetworkXNoPath:
            return None

    def get_nx_graph(self):

        g = self.graph.copy()
        labels = {i: str(i) for i in g.nodes}
        g = nx.relabel_nodes(g, labels)
        return g
