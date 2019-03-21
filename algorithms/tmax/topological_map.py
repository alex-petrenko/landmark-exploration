import math
from collections import deque
from hashlib import sha1

from utils.utils import log, ensure_contigious


def hash_observation(o):
    """Not the fastest way to do it, but plenty fast enough for our purposes."""
    o = ensure_contigious(o)
    return sha1(o).hexdigest()


class TopologicalMap:
    def __init__(self, initial_obs, directed_graph, info=None, verbose=False):
        self._verbose = verbose
        self._directed_graph = directed_graph

        self.landmarks = self.hashes = self.adjacency = self.positions = None
        self.curr_landmark_idx = 0

        self.edge_success = {}

        # variables needed for online localization
        self.new_landmark_candidate_frames = 0
        self.closest_landmarks = []

        self.reset(initial_obs, info)

    def reset(self, obs, info=None):
        """Create the graph with only one vertex."""
        self.landmarks = [obs]
        self.hashes = [hash_observation(obs)]

        self.positions = []
        if info is not None:
            self.add_position_info(info)

        self.adjacency = [[]]  # initial vertex has no neighbors
        self.curr_landmark_idx = 0

        self.new_episode()

    def new_episode(self):
        self.new_landmark_candidate_frames = 0
        self.closest_landmarks = []

        self._prune()

    def _log_verbose(self, msg, *args):
        if not self._verbose:
            return
        log.debug(msg, *args)

    @property
    def curr_landmark(self):
        return self.landmarks[self.curr_landmark_idx]

    def neighbor_indices(self):
        neighbors = [self.curr_landmark_idx]
        neighbors.extend(self.adjacency[self.curr_landmark_idx])
        return neighbors

    def reachable_indices(self, start_idx):
        """Run BFS from current landmark to find the list of landmarks reachable from the current landmark."""
        q = deque([])
        q.append(start_idx)
        reachable = {start_idx}  # hash set of visited vertices

        while len(q) > 0:
            curr_idx = q.popleft()
            for adj_idx in self.adjacency[curr_idx]:
                if adj_idx in reachable:
                    continue
                reachable.add(adj_idx)
                q.append(adj_idx)

        return list(reachable)

    def non_neighbor_indices(self):
        neighbors = self.neighbor_indices()
        non_neighbors = [i for i in range(len(self.landmarks)) if i not in neighbors]
        return non_neighbors

    def _add_directed_edge(self, i1, i2):
        if i2 not in self.adjacency[i1]:
            self.adjacency[i1].append(i2)
            self.edge_success[(i1, i2)] = 0.5
        self._log_verbose('New dir. edge %d-%d', i1, i2)

    def _add_undirected_edge(self, i1, i2):
        if i2 not in self.adjacency[i1]:
            self.adjacency[i1].append(i2)
            self.edge_success[(i1, i2)] = 0.5
        if i1 not in self.adjacency[i2]:
            self.adjacency[i2].append(i1)
            self.edge_success[(i2, i1)] = 0.5
        self._log_verbose('New und. edge %d-%d', i1, i2)

    def _add_edge(self, i1, i2):
        if self._directed_graph:
            self._add_directed_edge(i1, i2)
        else:
            self._add_undirected_edge(i1, i2)

    def set_curr_landmark(self, landmark_idx):
        """Replace current landmark with the given landmark. Create necessary edges if needed."""

        if landmark_idx == self.curr_landmark_idx:
            return

        if landmark_idx not in self.adjacency[self.curr_landmark_idx]:
            # create new edges, we found a loop closure!
            self._add_edge(self.curr_landmark_idx, landmark_idx)

        self._log_verbose('Change current landmark to %d', landmark_idx)
        self.curr_landmark_idx = landmark_idx

    def add_landmark(self, obs, info=None):
        new_landmark_idx = len(self.landmarks)
        self.landmarks.append(obs)
        self.hashes.append(hash_observation(obs))
        if info is not None:
            self.add_position_info(info)
        self.adjacency.append([])
        self._add_edge(self.curr_landmark_idx, new_landmark_idx)

        assert len(self.adjacency) == len(self.landmarks)
        assert len(self.positions) == len(self.landmarks)
        self._log_verbose('Added new landmark %d', new_landmark_idx)
        return new_landmark_idx

    def add_position_info(self, info):
        agent_pos = info.get('pos')
        if agent_pos is not None:
            agent_pos = [agent_pos['agent_x'], agent_pos['agent_y'], agent_pos['agent_a']]
        self.positions.append(agent_pos)

    def remove_edge(self, i1, i2):
        self.adjacency[i1].remove(i2)
        del self.edge_success[(i1, i2)]

    def num_edges(self):
        """Helper function for summaries."""
        num_edges = sum([len(adj) for adj in self.adjacency])
        return num_edges

    def update_edge_traversal(self, i1, i2, success):
        prev_value = self.edge_success[(i1, i2)]
        self.edge_success[(i1, i2)] = 0.5 * (prev_value + success)

    def _prune(self, threshold=0.1):
        """Remove edges with very low weight of traversal success."""
        remove = []
        for i, adj in enumerate(self.adjacency):
            for j in adj:
                if self.edge_success[(i, j)] <= threshold:
                    remove.append((i, j))

        if len(remove) > 0:
            log.info('Prune: removing edges %r', remove)
            for i1, i2 in remove:
                self.remove_edge(i1, i2)

    def _edge_weight(self, i1, i2):
        return -math.log(self.edge_success[(i1, i2)])

    def shortest_paths(self, idx):
        distances = [math.inf] * len(self.landmarks)
        distances[idx] = 0
        previous = [None] * len(self.landmarks)
        unvisited = list(range(len(self.landmarks)))

        while unvisited:
            u = min(unvisited, key=lambda node: distances[node])
            unvisited.remove(u)
            for neighbor in self.adjacency[u]:
                this_distance = distances[u] + self._edge_weight(u, neighbor)  # distance between each node is 1
                if this_distance < distances[neighbor]:
                    distances[neighbor] = this_distance
                    previous[neighbor] = u

        return distances, previous

    def get_path(self, from_idx, to_idx):
        path_lengths, path_prev = self.shortest_paths(from_idx)
        if path_prev[to_idx] is None:
            return None

        path = [to_idx]
        while path[-1] != from_idx:
            path.append(path_prev[path[-1]])

        return list(reversed(path))

    def to_nx_graph(self):
        import networkx as nx
        graph = nx.DiGraph()
        for i in range(len(self.landmarks)):
            pos = self.positions[i]
            graph.add_node(i, pos=(pos[0], pos[1]))
        for u, edges in enumerate(self.adjacency):
            for v in edges:
                graph.add_edge(u, v)
        return graph
