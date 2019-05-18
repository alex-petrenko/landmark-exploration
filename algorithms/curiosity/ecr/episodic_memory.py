import random

import numpy as np


class EpisodicMemory:
    """Contains the episodic memory, a buffer of frequently-updated observations."""

    class Landmark:
        def __init__(self, embedding, info):
            self.embedding = embedding
            self.info = info

    def __init__(self, params, embedding=None, info=None):
        self._counter = 0
        self._landmarks = {}
        self._previous_closest = set()

        self.params = params

        if embedding is not None and info is not None:
            self.add(embedding, info)

    def add(self, embedding, info):
        self._trim()

        new_landmark_idx = self._counter
        self._counter += 1

        self._landmarks[new_landmark_idx] = EpisodicMemory.Landmark(embedding, info)
        self._previous_closest.add(new_landmark_idx)

    def reset(self, embedding=None, info=None):
        self._landmarks = {}
        self._previous_closest = set()
        if embedding is not None and info is not None:
            self.add(embedding, info)

    def _trim(self):
        if len(self) >= self.params.episodic_memory_size:
            indices = list(self._landmarks.keys())
            while len(self) >= self.params.episodic_memory_size:
                while True:
                    landmark_to_delete = random.choice(indices)
                    if landmark_to_delete in self._previous_closest:
                        # this should not happen very often, unless percentile is close to 100
                        continue
                    else:
                        del self._landmarks[landmark_to_delete]
                        break

    def sample_landmarks(self, num_to_sample):
        num_to_sample = min(num_to_sample, len(self))

        embeddings, indices = [], []

        # first we always choose landmarks that were close during the previous frame
        for idx in self._previous_closest:
            embeddings.append(self._landmarks[idx].embedding)
            indices.append(idx)

        # fill the rest with random landmarks from memory
        random_indices = np.random.choice(list(self._landmarks.keys()), size=num_to_sample, replace=False)
        for idx in random_indices:
            if idx in self._previous_closest:
                continue

            embeddings.append(self._landmarks[idx].embedding)
            indices.append(idx)
            if len(embeddings) >= num_to_sample:
                break

        return embeddings[:num_to_sample], indices[:num_to_sample]

    def distance_percentile(self, distances, sample_indices, percentile):
        num_closest = max(1, int(len(sample_indices) * percentile / 100))

        assert len(distances) == len(sample_indices)

        # sort memory sample according to distance
        sorted_memory = sorted(zip(distances, sample_indices))

        sorted_distances, sorted_indices = list(zip(*sorted_memory))
        self._previous_closest = set(sorted_indices[:len(sorted_indices) // 2])

        return sorted_distances[num_closest - 1]

    @property
    def landmarks(self):
        return self._landmarks

    def __len__(self):
        return len(self._landmarks)
