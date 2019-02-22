import numpy as np
import tensorflow as tf

from algorithms.encoders import make_encoder
from algorithms.env_wrappers import get_observation_space
from algorithms.tf_utils import dense, placeholders_from_spaces
from algorithms.tmax.tmax_utils import TmaxMode
from utils.utils import log


class ReachabilityNetwork:
    def __init__(self, env, params):
        obs_space = get_observation_space(env)
        self.ph_obs_first, self.ph_obs_second = placeholders_from_spaces(obs_space, obs_space)
        self.ph_labels = tf.placeholder(dtype=tf.int32, shape=(None, ))

        with tf.variable_scope('reach'):
            encoder = tf.make_template(
                'siamese_enc', make_encoder, create_scope_now_=True, env=env, regularizer=None, params=params,
            )

            obs_first_enc = encoder(self.ph_obs_first)
            obs_second_enc = encoder(self.ph_obs_second)
            observations_encoded = tf.concat([obs_first_enc.encoded_input, obs_second_enc.encoded_input], axis=1)

            fc_layers = [256, 256]
            x = observations_encoded
            for fc_layer_size in fc_layers:
                x = dense(x, fc_layer_size)

            # embedding = obs_first_enc.encoded_input
            # self.obs_decoded = DecoderCNN(embedding, 'reach_dec').decoded

            logits = tf.contrib.layers.fully_connected(x, 2, activation_fn=None)
            self.probabilities = tf.nn.softmax(logits)

            self.reach_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.ph_labels)
            self.reach_loss = tf.reduce_mean(self.reach_loss)

            # self.normalized_obs = obs_first_enc.normalized_obs
            # # self.normalized_obs = tf.zeros_like(obs_first_enc.normalized_obs)
            # self.reconst_loss = tf.nn.l2_loss(self.normalized_obs - self.obs_decoded)

            self.loss = self.reach_loss

    def get_probabilities(self, session, obs_first, obs_second):
        probabilities = session.run(
            self.probabilities,
            feed_dict={self.ph_obs_first: obs_first, self.ph_obs_second: obs_second},
        )
        return probabilities

    def distances(self, session, obs_first, obs_second):
        probs = self.get_probabilities(session, obs_first, obs_second)
        return [p[1] for p in probs]


class ReachabilityBuffer:
    """Training data for the reachability network (observation pairs and labels)."""

    def __init__(self, params):
        self.obs_first, self.obs_second, self.labels = [], [], []
        self.params = params

    def extract_data(self, trajectories):
        close = self.params.reachable_threshold
        far = self.params.unreachable_threshold

        obs_first_close, obs_second_close, labels_close = [], [], []
        obs_first_far, obs_second_far, labels_far = [], [], []

        for trajectory in trajectories:
            if len(trajectory) <= 1:
                continue

            obs = trajectory.obs

            deliberate = [int(trajectory.deliberate_action[0])] * len(trajectory)
            for i in range(1, len(trajectory)):
                deliberate[i] = deliberate[i - 1]
                if trajectory.deliberate_action[i]:
                    deliberate[i] += 1

            close_idx, far_idx = 0, 0
            for i in range(len(trajectory)):
                if trajectory.modes[i] != TmaxMode.EXPLORATION:
                    continue

                # everything between i and close_idx is considered "close"
                while close_idx < len(trajectory) and deliberate[close_idx] - deliberate[i] < close:
                    close_idx += 1

                # everything between far_idx and len(trajectory) is considered "far"
                while far_idx < len(trajectory) and deliberate[far_idx] - deliberate[i] < far:
                    far_idx += 1

                assert far_idx >= close_idx

                # for each frame sample one "close" and one "far" example
                # first: sample "close" example
                second_idx = np.random.randint(i, close_idx)
                obs_first_close.append(obs[i])
                obs_second_close.append(obs[second_idx])
                labels_close.append(0)

                # sample "far" example
                if len(trajectory) - far_idx > 0:
                    second_idx = np.random.randint(far_idx, len(trajectory))
                    obs_first_far.append(obs[i])
                    obs_second_far.append(obs[second_idx])
                    labels_far.append(1)

        assert len(obs_first_close) == len(obs_second_close)
        assert len(obs_first_close) == len(labels_close)
        assert len(obs_first_far) == len(obs_second_far)
        assert len(obs_first_far) == len(labels_far)

        num_examples = min(len(obs_first_close), len(obs_first_far))
        obs_first = obs_first_close[:num_examples] + obs_first_far[:num_examples]
        obs_second = obs_second_close[:num_examples] + obs_second_far[:num_examples]
        labels = labels_close[:num_examples] + labels_far[:num_examples]

        if len(self.obs_first) <= 0:
            self.obs_first = np.array(obs_first)
            self.obs_second = np.array(obs_second)
            self.labels = np.array(labels, dtype=np.int32)
        elif len(obs_first) > 0:
            self.obs_first = np.append(self.obs_first, obs_first, axis=0)
            self.obs_second = np.append(self.obs_second, obs_second, axis=0)
            self.labels = np.append(self.labels, labels, axis=0)

        self._discard_data()

        assert len(self.obs_first) == len(self.obs_second)
        assert len(self.obs_first) == len(self.labels)

    def _discard_data(self):
        """Remove some data if the current buffer is too big."""
        target_size = self.params.reachability_target_buffer_size
        if len(self.obs_first) <= target_size:
            return

        self.shuffle_data()
        self.obs_first = self.obs_first[:target_size]
        self.obs_second = self.obs_second[:target_size]
        self.labels = self.labels[:target_size]

    def has_enough_data(self):
        len_data, min_data = len(self.obs_first), self.params.reachability_target_buffer_size // 10
        if len_data < min_data:
            log.info('Need to gather more data to train reachability net, %d/%d', len_data, min_data)
            return False
        return True

    def shuffle_data(self):
        if len(self.obs_first) <= 0:
            return

        chaos = np.random.permutation(len(self.obs_first))
        self.obs_first = self.obs_first[chaos]
        self.obs_second = self.obs_second[chaos]
        self.labels = self.labels[chaos]
