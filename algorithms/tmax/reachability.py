import numpy as np
import tensorflow as tf

from algorithms.encoders import make_encoder
from algorithms.env_wrappers import get_observation_space
from algorithms.tf_utils import dense, placeholders_from_spaces
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

            obs_first_enc = encoder(self.ph_obs_first).encoded_input
            obs_second_enc = encoder(self.ph_obs_second).encoded_input
            observations_encoded = tf.concat([obs_first_enc, obs_second_enc], axis=1)

            fc_layers = [256, 256]
            x = observations_encoded
            for fc_layer_size in fc_layers:
                x = dense(x, fc_layer_size)

            logits = tf.contrib.layers.fully_connected(x, 2, activation_fn=None)
            self.probabilities = tf.nn.softmax(logits)

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.ph_labels)
            self.loss = tf.reduce_mean(self.loss)

    def get_probabilities(self, session, obs_first, obs_second):
        probabilities = session.run(
            self.probabilities,
            feed_dict={self.ph_obs_first: obs_first, self.ph_obs_second: obs_second},
        )
        return probabilities

    def get_reachability(self, session, obs_first, obs_second):
        probs = self.get_probabilities(session, obs_first, obs_second)
        return [p[1] for p in probs]


class ReachabilityBuffer:
    """Training data for the reachability network (observation pairs and labels)."""

    def __init__(self, params):
        self.obs_first, self.obs_second, self.labels = [], [], []
        self.params = params

    def extract_data(self, trajectories, bootstrap_period):
        obs_first, obs_second, labels = [], [], []
        total_num_reachable = total_num_unreachable = 0

        for trajectory in trajectories:
            obs = trajectory.obs
            episode_len = len(obs)

            obs_pairs_fraction = self.params.obs_pairs_per_episode if bootstrap_period else 1.0
            num_obs_pairs = int(obs_pairs_fraction * episode_len)

            reachable_thr = self.params.reachable_threshold
            unreachable_thr = self.params.unreachable_threshold

            num_reachable = num_unreachable = attempt = 0

            try:
                while num_reachable + num_unreachable < num_obs_pairs and attempt < 3 * num_obs_pairs:
                    # some attempts to sample a training pair might fail, we want to account for that
                    attempt += 1

                    # determine if we want a reachable pair or not
                    reachable = total_num_reachable <= total_num_unreachable
                    threshold = reachable_thr if reachable else unreachable_thr

                    # sample first obs in a pair
                    first_idx = np.random.randint(0, episode_len - threshold - 1)

                    # sample second obs
                    if reachable:
                        second_idx = np.random.randint(first_idx, first_idx + reachable_thr)
                    else:
                        second_idx = np.random.randint(first_idx + unreachable_thr, episode_len)
                        if not bootstrap_period:
                            if trajectory.current_landmark_idx[second_idx] in trajectory.neighbor_indices[first_idx]:
                                # selected "unreachable" observation is actually in the graph neighborhood of
                                # the first observation, so skip this pair
                                # log.info('Skipped unreachable pair %d %d', first_idx, second_idx)
                                continue

                    obs_first.append(obs[first_idx])
                    obs_second.append(obs[second_idx])
                    labels.append(int(reachable))

                    if reachable:
                        num_reachable += 1
                        total_num_reachable += 1
                    else:
                        num_unreachable += 1
                        total_num_unreachable += 1
            except ValueError:
                # just in case, if some episode is e.g. too short for unreachable pair
                log.exception(f'Value error in Reachability buffer! Episode len {episode_len}')

        if len(obs_first) <= 0:
            return

        if len(self.obs_first) <= 0:
            self.obs_first = np.array(obs_first)
            self.obs_second = np.array(obs_second)
            self.labels = np.array(labels, dtype=np.int32)
        else:
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
        len_data, min_data = len(self.obs_first), self.params.reachability_target_buffer_size // 2
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
