import random

import numpy as np


class EpisodicMemory:
    """Contains the episodic memory, a buffer of frequently-updated observations."""

    def __init__(self, params, embedding=None):
        self.arr = []
        self.batch_num = 0

        self.params = params

        if embedding is not None:
            self.add(embedding)

    def add(self, embedding):
        self._trim()
        self.arr.append(embedding)

    def distances(self, session, reachability, embedding):
        dist = []
        for emb in self.arr:
            dist.append(reachability.distances(session, embedding, emb))
        assert len(dist) == len(self.arr)
        return np.percentile(dist, 90)

    def reset(self, embedding=None):
        self.arr.clear()
        if embedding is not None:
            self.add(embedding)

    def _trim(self):
        if len(self.arr) >= self.params.episodic_memory_size:
            random.shuffle(self.arr)
            while len(self.arr) >= self.params.episodic_memory_size:
                self.arr.pop()

    def __len__(self):
        return len(self.arr)
