class ObservationEncoder:
    """
    Turn landmark observations into vectors in a lazy way (only when they are needed).
    Uses hashes to determine the identity of the observation, so the exact same observation won't be encoded twice.
    """

    def __init__(self, encode_func):
        self.encoded_obs = {}  # map from observation hash to embedding vector
        self.encode_func = encode_func

    def reset(self):
        """Discard all previous embeddings (e.g. after an interation of training)."""
        self.encoded_obs = {}

    def encode(self, session, landmark_obs, landmark_hashes):
        landmarks_to_encode, hashes_to_encode = [], []

        for i, landmark_hash in enumerate(landmark_hashes):
            if landmark_hash not in self.encoded_obs:
                landmarks_to_encode.append(landmark_obs[i])
                hashes_to_encode.append(landmark_hashes[i])

        if len(landmarks_to_encode) > 0:
            encoded = self.encode_func(session, landmarks_to_encode)
            assert len(encoded) == len(hashes_to_encode)

            for i in range(len(encoded)):
                self.encoded_obs[hashes_to_encode[i]] = encoded[i]
