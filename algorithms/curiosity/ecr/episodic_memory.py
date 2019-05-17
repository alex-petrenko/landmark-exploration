import random


class EpisodicMemory:
    """Contains the episodic memory, a buffer of frequently-updated observations."""

    class Landmark:
        def __init__(self, embedding, info):
            self.embedding = embedding
            self.info = info

    def __init__(self, params, embedding=None, info=None):
        self._arr = []
        self.batch_num = 0
        self.params = params

        if embedding is not None and info is not None:
            self.add(embedding, info)

    def add(self, embedding, info):
        self._trim()
        self.arr.append(EpisodicMemory.Landmark(embedding, info))

    def reset(self, embedding=None, info=None):
        self.arr.clear()
        if embedding is not None and info is not None:
            self.add(embedding, info)

    def _trim(self):
        if len(self.arr) >= self.params.episodic_memory_size:
            random.shuffle(self.arr)

            while len(self.arr) >= self.params.episodic_memory_size:
                self.arr.pop()

    @property
    def arr(self):
        return self._arr

    @property
    def embeddings(self):
        return [lm.embedding for lm in self.arr]

    def __len__(self):
        return len(self.arr)
