import random

import numpy as np


class EpisodicMemory:
    """Contains the episodic memory, a buffer of frequently-updated observations."""

    def __init__(self, params, embedding=None):
        self._arr = []
        self.batch_num = 0

        self.params = params

        if embedding is not None:
            self.add(embedding)

    def add(self, embedding):
        self._trim()
        self.arr.append(embedding)

    def reset(self, embedding=None):
        self.arr.clear()
        if embedding is not None:
            self.add(embedding)

    def _trim(self):
        if len(self.arr) >= self.params.episodic_memory_size:
            random.shuffle(self.arr)
            while len(self.arr) >= self.params.episodic_memory_size:
                self.arr.pop()

    @property
    def arr(self):
        return self._arr

    def __len__(self):
        return len(self.arr)
