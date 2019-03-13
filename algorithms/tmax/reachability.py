import os
import shutil
import time
from collections import deque
from os.path import join

import cv2
import numpy as np
import tensorflow as tf

from algorithms.buffer import Buffer
from algorithms.encoders import make_encoder
from algorithms.env_wrappers import get_observation_space
from algorithms.tf_utils import dense, placeholders_from_spaces
from utils.timing import Timing
from utils.utils import log, vis_dir, ensure_dir_exists


class ReachabilityNetwork:
    def __init__(self, env, params):
        obs_space = get_observation_space(env)
        self.ph_obs_first, self.ph_obs_second = placeholders_from_spaces(obs_space, obs_space)
        self.ph_labels = tf.placeholder(dtype=tf.int32, shape=(None, ))

        with tf.variable_scope('reach') as scope:
            reg = tf.contrib.layers.l2_regularizer(scale=1e-5)

            encoder = tf.make_template(
                'siamese_enc', make_encoder, create_scope_now_=True, env=env, regularizer=reg, params=params,
            )

            obs_first_enc = encoder(self.ph_obs_first)
            obs_second_enc = encoder(self.ph_obs_second)
            observations_encoded = tf.concat([obs_first_enc.encoded_input, obs_second_enc.encoded_input], axis=1)

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
        self.buffer = Buffer()
        self.close_buff, self.far_buff = Buffer(), Buffer()
        self.batch_num = 0

        self._vis_dirs = deque([])

        self.params = params

    def extract_data(self, trajectories):
        timing = Timing()

        close, far = self.params.reachable_threshold, self.params.unreachable_threshold

        with timing.timeit('trajectories'):
            data = Buffer()
            for trajectory in trajectories:
                obs = trajectory.obs

                indices = list(range(len(trajectory)))
                np.random.shuffle(indices)

                for i in indices:
                    if len(data) > self.params.reachability_target_buffer_size // 5:
                        break

                    close_i = min(i + close, len(trajectory))
                    far_i = min(i + far, len(trajectory))

                    # sample close observation pair
                    second_idx = np.random.randint(i, close_i)
                    data.add(obs_first=obs[i], obs_second=obs[second_idx], labels=0)

                    # sample far observation pair
                    if far_i < len(trajectory):
                        second_idx = np.random.randint(far_i, len(trajectory))
                        data.add(obs_first=obs[i], obs_second=obs[second_idx], labels=1)

        with timing.timeit('add_and_shuffle'):
            self.buffer.add_buff(data)
            self.shuffle_data()
            self.buffer.trim_at(self.params.reachability_target_buffer_size)

        if self.batch_num % 20 == 0:
            with timing.timeit('visualize'):
                self._visualize_data()

        self.batch_num += 1
        log.info('Reachability timing %s', timing)

    def has_enough_data(self):
        len_data, min_data = len(self.buffer), self.params.reachability_target_buffer_size // 40
        if len_data < min_data:
            log.info('Need to gather more data to train reachability net, %d/%d', len_data, min_data)
            return False
        return True

    def shuffle_data(self):
        self.buffer.shuffle_data()

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
        img_folder = join(img_folder, f'reach_{time.time()}')
        ensure_dir_exists(img_folder)

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
