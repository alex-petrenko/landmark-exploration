from hashlib import sha1

import numpy as np

from utils.utils import log


def hash_observation(o):
    """Not the fastest way to do it, but plenty fast enough for our purposes."""
    if not o.flags['C_CONTIGUOUS']:
        o = np.ascontiguousarray(o)
    return sha1(o).hexdigest()


class TopologicalMap:
    def __init__(self, initial_obs, verbose=False):
        self.verbose = verbose
        self.landmarks = self.hashes = self.adjacency = None
        self.curr_landmark_idx = 0
        self.reset(initial_obs)

    def reset(self, obs):
        """Create the graph with only one vertex."""
        self.landmarks = [obs]
        self.hashes = [hash_observation(obs)]
        self.adjacency = [[]]  # initial vertex has no neighbors
        self.curr_landmark_idx = 0

    def _log_verbose(self, msg, *args):
        if not self.verbose:
            return
        log.debug(msg, *args)

    @property
    def curr_landmark(self):
        return self.landmarks[self.curr_landmark_idx]

    def neighbor_indices(self):
        neighbors = [self.curr_landmark_idx]
        neighbors.extend([i for i in self.adjacency[self.curr_landmark_idx]])
        return neighbors

    def non_neighbor_indices(self):
        neighbors = self.neighbor_indices()
        non_neighbors = [i for i in range(len(self.landmarks)) if i not in neighbors]
        return non_neighbors

    def _add_directed_edge(self, i1, i2):
        self.adjacency[i1].append(i2)
        self._log_verbose('New dir. edge %d-%d', i1, i2)

    def _add_undirected_edge(self, i1, i2):
        self.adjacency[i1].append(i2)
        self.adjacency[i2].append(i1)
        self._log_verbose('New und. edge %d-%d', i1, i2)

    def set_curr_landmark(self, landmark_idx):
        """Replace current landmark with the given landmark. Create necessary edges if needed."""

        if landmark_idx == self.curr_landmark_idx:
            return

        if landmark_idx not in self.adjacency[self.curr_landmark_idx]:
            # create new edges, we found a loop closure!
            assert self.curr_landmark_idx not in self.adjacency[landmark_idx]
            self._add_undirected_edge(self.curr_landmark_idx, landmark_idx)

        self._log_verbose('Change current landmark to %d', landmark_idx)
        self.curr_landmark_idx = landmark_idx

    def add_landmark(self, obs):
        new_landmark_idx = len(self.landmarks)
        self.landmarks.append(obs)
        self.hashes.append(hash_observation(obs))

        self.adjacency.append([])
        self._add_undirected_edge(self.curr_landmark_idx, new_landmark_idx)
        assert len(self.adjacency) == len(self.landmarks)
        self._log_verbose('Added new landmark %d', new_landmark_idx)
        return new_landmark_idx

    def num_undirected_edges(self):
        """Helper function for summaries."""
        num_edges = [len(adj) for adj in self.adjacency]
        num_edges = sum(num_edges) / 2
        return num_edges

    @property
    def nx_graph(self):
        import networkx as nx
        graph = nx.DiGraph()
        for i in range(len(self.landmarks)):
            graph.add_node(i)
        for u, edges in enumerate(self.adjacency):
            for v in edges:
                graph.add_edge(u, v)
        return graph
