class LandmarksEncoder:
    """
    Turn landmark observations into vectors in a lazy way (only when they are needed).
    Uses hashes to determine the identity of the observation, so the exact same observation won't be encoded twice.
    """

    def __init__(self):
        self.encoded_landmarks = {}  # map from observation hash to embedding vector
        self.encode_func = None

    def reset(self, encode_func):
        """Discard all previous embeddings (e.g. after an interation of training)."""
        self.encoded_landmarks = {}
        self.encode_func = encode_func

    def encode(self, landmark_obs, landmark_hashes):
        landmarks_to_encode, hashes_to_encode = [], []

        for i, landmark_hash in enumerate(landmark_hashes):
            if landmark_hash not in self.encoded_landmarks:
                landmarks_to_encode.append(landmark_obs[i])
                hashes_to_encode.append(landmark_hashes[i])

        if len(landmarks_to_encode) > 0:
            encoded = self.encode_func(landmarks_to_encode)
            assert len(encoded) == len(hashes_to_encode)

            for i in range(len(encoded)):
                self.encoded_landmarks[hashes_to_encode[i]] = encoded[i]
