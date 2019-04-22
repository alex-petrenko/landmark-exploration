import os
import random
import time
import shutil
from collections import deque
from os.path import join

import cv2
import numpy as np
import tensorflow as tf
from keras import Input, Model
from keras.layers import Lambda, concatenate

from algorithms.buffer import Buffer
from algorithms.reachability.observation_encoder import ObservationEncoder
from algorithms.encoders import make_encoder
from algorithms.env_wrappers import main_observation_space
from algorithms.resnet_keras import ResnetBuilder, _top_network
from algorithms.tf_utils import dense, placeholders_from_spaces
from algorithms.topological_maps.topological_map import hash_observation
from utils.timing import Timing
from utils.utils import log, vis_dir, ensure_dir_exists


class ReachabilityNetwork:
    def __init__(self, env, params):
        obs_space = main_observation_space(env)
        self.ph_obs_first, self.ph_obs_second = placeholders_from_spaces(obs_space, obs_space)
        self.ph_labels = tf.placeholder(dtype=tf.int32, shape=(None,))

        with tf.variable_scope('reach') as scope:
            reg = tf.contrib.layers.l2_regularizer(scale=1e-5)

            encoder = tf.make_template(
                'siamese_enc', make_encoder, create_scope_now_=True,
                obs_space=obs_space, regularizer=reg, params=params,
            )

            obs_first_enc = encoder(self.ph_obs_first)
            obs_second_enc = encoder(self.ph_obs_second)

            self.first_encoded = obs_first_enc.encoded_input
            self.second_encoded = obs_second_enc.encoded_input

            observations_encoded = tf.concat([self.first_encoded, self.second_encoded], axis=1)

            fc_layers = [256, 256]
            x = observations_encoded
            for fc_layer_size in fc_layers:
                x = dense(x, fc_layer_size, reg)

            logits = tf.contrib.layers.fully_connected(x, 2, activation_fn=None)
            self.probabilities = tf.nn.softmax(logits)
            self.correct = tf.reduce_mean(
                tf.to_float(tf.equal(self.ph_labels, tf.cast(tf.argmax(logits, axis=1), tf.int32))),
            )

            self.reach_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.ph_labels)
            self.reach_loss = tf.reduce_mean(self.reach_loss)

            reg_losses = tf.losses.get_regularization_losses(scope=scope.name)
            self.reg_loss = tf.reduce_sum(reg_losses)

            self.loss = self.reach_loss + self.reg_loss

            # helpers to encode observations (saves time)
            # does not matter if we use first vs second here
            self.ph_obs = self.ph_obs_first
            self.encoded_observation = self.first_encoded

        # other stuff not related to computation graph
        self.obs_encoder = ObservationEncoder(encode_func=self.encode_observation)

    def get_probabilities(self, session, obs_first_encoded, obs_second_encoded):
        assert len(obs_first_encoded) == len(obs_second_encoded)
        if len(obs_first_encoded) <= 0:
            return []

        probabilities = session.run(
            self.probabilities,
            feed_dict={self.first_encoded: obs_first_encoded, self.second_encoded: obs_second_encoded},
        )
        return probabilities

    def distances(self, session, obs_first_encoded, obs_second_encoded):
        probs = self.get_probabilities(session, obs_first_encoded, obs_second_encoded)
        return [p[1] for p in probs]

    def distances_from_obs(self, session, obs_first, obs_second, hashes_first=None, hashes_second=None):
        """Use encoder to get embedding vectors first."""
        obs_encoder = self.obs_encoder

        if hashes_first is None:
            hashes_first = [hash_observation(obs) for obs in obs_first]
        if hashes_second is None:
            hashes_second = [hash_observation(obs) for obs in obs_second]

        obs_encoder.encode(session, obs_first + obs_second, hashes_first + hashes_second)

        obs_first_encoded = [obs_encoder.encoded_obs[h] for h in hashes_first]
        obs_second_encoded = [obs_encoder.encoded_obs[h] for h in hashes_second]

        d = self.distances(session, obs_first_encoded, obs_second_encoded)
        return d

    def encode_observation(self, session, obs):
        return session.run(self.encoded_observation, feed_dict={self.ph_obs: obs})


class ReachabilityNetworkResnet:
    def __init__(self, env, params):
        width, height, channels = main_observation_space(env)
        size_embedding = 512

        with tf.variable_scope('reach'):
            branch = ResnetBuilder.build_resnet_18([channels, height, width], size_embedding, is_classification=False)

            obs_first = Input(shape=(height, width, channels))
            obs_second = Input(shape=(height, width, channels))

            # sharing weights
            self.first_encoded = branch(Lambda(lambda x_: x_[:, :, :, :channels])(obs_first))
            self.second_encoded = branch(Lambda(lambda x_: x_[:, :, :, channels:])(obs_second))

            raw_result = concatenate([self.first_encoded, self.second_encoded])
            self.probabilities = _top_network(raw_result)

            self.model = Model(inputs=[obs_first, obs_second], outputs=self.probabilities)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, name='distance_opt')
            self.model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

            # helpers to encode observations (saves time)
            # does not matter if we use first vs second here
            self.ph_obs = obs_first
            self.encoded_observation = self.first_encoded

        # other stuff not related to computation graph
        self.obs_encoder = ObservationEncoder(encode_func=self.encode_observation)

    def get_probabilities(self, session, obs_first_encoded, obs_second_encoded):
        assert len(obs_first_encoded) == len(obs_second_encoded)
        if len(obs_first_encoded) <= 0:
            return []

        probabilities = session.run(
            self.probabilities,
            feed_dict={self.first_encoded: obs_first_encoded, self.second_encoded: obs_second_encoded},
        )
        return probabilities

    def distances(self, session, obs_first_encoded, obs_second_encoded):
        probs = self.get_probabilities(session, obs_first_encoded, obs_second_encoded)
        return [p[1] for p in probs]

    def distances_from_obs(self, session, obs_first, obs_second, hashes_first=None, hashes_second=None):
        """Use encoder to get embedding vectors first."""
        obs_encoder = self.obs_encoder

        if hashes_first is None:
            hashes_first = [hash_observation(obs) for obs in obs_first]
        if hashes_second is None:
            hashes_second = [hash_observation(obs) for obs in obs_second]

        obs_encoder.encode(session, obs_first + obs_second, hashes_first + hashes_second)

        obs_first_encoded = [obs_encoder.encoded_obs[h] for h in hashes_first]
        obs_second_encoded = [obs_encoder.encoded_obs[h] for h in hashes_second]

        d = self.distances(session, obs_first_encoded, obs_second_encoded)
        return d

    def encode_observation(self, session, obs):
        return session.run(self.encoded_observation, feed_dict={self.ph_obs: obs})

    def train(self, agent, buffer, params):
        summary = None
        prev_loss = 1e10
        num_epochs = params.reachability_train_epochs
        batch_size = params.reachability_batch_size

        log.info('Training reachability %d pairs, batch %d, epochs %d', len(buffer), batch_size, num_epochs)

        obs_first, obs_second, labels = buffer.obs_first, buffer.obs_second, buffer.labels




class ReachabilityBuffer:
    """Training data for the reachability network (observation pairs and labels)."""

    def __init__(self, params):
        self.buffer = Buffer()
        self.close_buff, self.far_buff = Buffer(), Buffer()
        self.batch_num = 0

        self._vis_dirs = deque([])

        self.params = params

    # noinspection PyMethodMayBeStatic,PyUnusedLocal
    def skip(self, trajectory, i):
        return False

    def extract_data(self, trajectories):
        timing = Timing()

        close, far = self.params.reachable_threshold, self.params.unreachable_threshold

        num_close, num_far = 0, 0
        data_added = 0

        with timing.timeit('trajectories'):
            for trajectory in trajectories:
                obs = trajectory.obs

                indices = list(range(len(trajectory)))
                np.random.shuffle(indices)

                for i in indices:
                    if data_added > self.params.reachability_target_buffer_size // 3:  # to limit memory usage
                        break

                    close_i = min(i + close, len(trajectory))
                    far_i = min(i + far, len(trajectory))

                    # sample close observation pair
                    first_idx = i
                    second_idx = np.random.randint(i, close_i)
                    if self.params.reachability_symmetric and random.random() < 0.5:
                        first_idx, second_idx = second_idx, first_idx

                    if not self.skip(trajectory, first_idx) and not self.skip(trajectory, second_idx):
                        self.buffer.add(obs_first=obs[first_idx], obs_second=obs[second_idx], labels=0)
                        data_added += 1
                        num_close += 1

                    # sample far observation pair
                    if far_i < len(trajectory):
                        first_idx = i
                        second_idx = np.random.randint(far_i, len(trajectory))
                        if self.params.reachability_symmetric and random.random() < 0.5:
                            first_idx, second_idx = second_idx, first_idx

                        if not self.skip(trajectory, first_idx) and not self.skip(trajectory, second_idx):
                            self.buffer.add(obs_first=obs[first_idx], obs_second=obs[second_idx], labels=1)
                            data_added += 1
                            num_far += 1

        with timing.timeit('shuffle'):
            # This is to avoid shuffling data every time. We grow the buffer a little more (up to 1.5 size of max
            # buffer) and shuffle and trim only then (or when we need it for training).
            # Adjust this 1.5 parameter for memory consumption.
            if len(self.buffer) > 1.5 * self.params.reachability_target_buffer_size:
                self.shuffle_data()
                self.buffer.trim_at(self.params.reachability_target_buffer_size)

        if self.batch_num % 20 == 0:
            with timing.timeit('visualize'):
                self._visualize_data()

        self.batch_num += 1
        log.info('num close %d, num far %d, reachability timing %s', num_close, num_far, timing)

    def has_enough_data(self):
        len_data, min_data = len(self.buffer), self.params.reachability_target_buffer_size // 40
        if len_data < min_data:
            log.info('Need to gather more data to train reachability net, %d/%d', len_data, min_data)
            return False
        return True

    def get_buffer(self):
        if len(self.buffer) > self.params.reachability_target_buffer_size:
            self.shuffle_data()
            self.buffer.trim_at(self.params.reachability_target_buffer_size)
        return self.buffer

    def shuffle_data(self):
        self.buffer.shuffle_data()

    def reset(self):
        self.buffer.clear()

    def _visualize_data(self):
        min_vis = 10
        if len(self.buffer) < min_vis:
            return

        close_examples, far_examples = [], []
        labels = self.buffer.labels
        obs_first, obs_second = self.buffer.obs_first, self.buffer.obs_second

        for i in range(len(labels)):
            if labels[i] == 0 and len(close_examples) < min_vis:
                close_examples.append((obs_first[i], obs_second[i]))
            elif labels[i] == 1 and len(far_examples) < min_vis:
                far_examples.append((obs_first[i], obs_second[i]))

        if len(close_examples) < min_vis or len(far_examples) < min_vis:
            return

        img_folder = vis_dir(self.params.experiment_dir())
        img_folder = ensure_dir_exists(join(img_folder, 'reach'))
        img_folder = ensure_dir_exists(join(img_folder, f'reach_{time.time()}'))

        def save_images(examples, close_or_far):
            for visualize_i in range(len(examples)):
                img_first_name = join(img_folder, f'{close_or_far}_{visualize_i}_first.png')
                img_second_name = join(img_folder, f'{close_or_far}_{visualize_i}_second.png')
                cv2.imwrite(img_first_name, examples[visualize_i][0])
                cv2.imwrite(img_second_name, examples[visualize_i][1])

        save_images(close_examples, 'close')
        save_images(far_examples, 'far')

        self._vis_dirs.append(img_folder)
        while len(self._vis_dirs) > 20:
            dir_name = self._vis_dirs.popleft()
            if os.path.isdir(dir_name):
                shutil.rmtree(dir_name)
