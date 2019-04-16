from algorithms.topological_maps.topological_map import hash_observation
from utils.utils import log


class ObservationEncoder:
    """
    Turn landmark observations into vectors in a lazy way (only when they are needed).
    Uses hashes to determine the identity of the observation, so the exact same observation won't be encoded twice.
    """

    def __init__(self, encode_func):
        self.encoded_obs = {}  # map from observation hash to embedding vector
        self.encode_func = encode_func
        self.size_limit = 500000  # max number of vectors to store
        self.max_batch = 1024  # to avoid GPU memory overflow

    def reset(self):
        """Discard all previous embeddings (e.g. after an interation of training)."""
        self.encoded_obs = {}

    def encode(self, session, landmark_obs, landmark_hashes=None):
        if landmark_hashes is None:
            landmark_hashes = [hash_observation(o) for o in landmark_obs]

        assert len(landmark_obs) == len(landmark_hashes)

        if len(self.encoded_obs) > self.size_limit:
            log.info(
                'Observation encoder reset due to size limit exceeded %d/%d', len(self.encoded_obs), self.size_limit,
            )
            self.reset()

        landmarks_to_encode, hashes_to_encode = [], []
        hashes_to_encode_set = set()

        for i, landmark_hash in enumerate(landmark_hashes):
            if landmark_hash not in self.encoded_obs and landmark_hash not in hashes_to_encode_set:
                landmarks_to_encode.append(landmark_obs[i])
                hashes_to_encode.append(landmark_hash)
                hashes_to_encode_set.add(landmark_hash)

        if len(landmarks_to_encode) > 0:
            encoded = []

            for i in range(0, len(landmarks_to_encode), self.max_batch):
                start, end = i, i + self.max_batch
                encoded_batch = self.encode_func(session, landmarks_to_encode[start:end])
                encoded.extend(encoded_batch)

            assert len(encoded) == len(hashes_to_encode)
            for i in range(len(encoded)):
                self.encoded_obs[hashes_to_encode[i]] = encoded[i]

        return self.encoded_obs
