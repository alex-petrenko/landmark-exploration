import os
import random
import shutil
import time
from collections import deque
from os.path import join

import cv2
import matplotlib.pyplot as plt
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

        with tf.variable_scope('reach'):
            encoder = tf.make_template(
                'siamese_enc', make_encoder, create_scope_now_=True, env=env, regularizer=None, params=params,
            )

            obs_first_enc = encoder(self.ph_obs_first)
            obs_second_enc = encoder(self.ph_obs_second)
            observations_encoded = tf.concat([obs_first_enc.encoded_input, obs_second_enc.encoded_input], axis=1)

            self.reg = tf.contrib.layers.l2_regularizer(scale=1e-10)

            fc_layers = [256, 256]
            conv_features = tf.stop_gradient(observations_encoded)
            x = conv_features
            for fc_layer_size in fc_layers:
                x = dense(x, fc_layer_size, self.reg)

            logits = tf.contrib.layers.fully_connected(x, 2, activation_fn=None)
            self.probabilities = tf.nn.softmax(logits)
            self.correct = tf.reduce_mean(tf.to_float(tf.equal(self.ph_labels, tf.cast(tf.argmax(logits, axis=1), tf.int32))))

            self.reach_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.ph_labels)
            self.reach_loss = tf.reduce_mean(self.reach_loss)

            x = tf.stop_gradient(obs_first_enc.encoded_input)
            x = dense(x, 256, self.reg)
            x = dense(x, 256, self.reg)
            first_logits = tf.contrib.layers.fully_connected(x, 2, activation_fn=None)
            self.first_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=first_logits, labels=self.ph_labels)
            self.first_loss = tf.reduce_mean(self.first_loss)
            self.first_correct = tf.reduce_mean(tf.to_float(tf.equal(self.ph_labels, tf.cast(tf.argmax(first_logits, axis=1), dtype=tf.int32))))

            x = tf.stop_gradient(obs_second_enc.encoded_input)
            x = dense(x, 256, self.reg)
            x = dense(x, 256, self.reg)
            second_logits = tf.contrib.layers.fully_connected(x, 2, activation_fn=None)
            self.second_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=second_logits, labels=self.ph_labels)
            self.second_loss = tf.reduce_mean(self.second_loss)
            self.second_correct = tf.reduce_mean(tf.to_float(tf.equal(self.ph_labels, tf.cast(tf.argmax(second_logits, axis=1), dtype=tf.int32))))

            # self.obs_decoded = DecoderCNN(tf.stop_gradient(obs_first_enc.encoded_input), 'reach_dec').decoded
            # self.normalized_obs = obs_first_enc.normalized_obs
            # self.reconst_loss = tf.nn.l2_loss(self.normalized_obs - self.obs_decoded) / (64 * 64)

            self.loss = self.reach_loss + self.first_loss + self.second_loss

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
        close = self.params.reachable_threshold
        far = self.params.unreachable_threshold

        timing = Timing()

        with timing.timeit('init'):
            close_buff, far_buff = Buffer(), Buffer()

            num_close, num_far = len(self.close_buff), len(self.far_buff)
            sum_dist_close, sum_dist_far = 0, 0
            avg_dist_close, avg_dist_far = 0, 0

            if num_close > 0:
                sum_dist_close = np.sum(self.close_buff.dist)
                avg_dist_close = sum_dist_close / num_close
            if num_far > 0:
                sum_dist_far = np.sum(self.far_buff.dist)
                avg_dist_far = sum_dist_far / num_far
            log.info(
                'Avg close %.3f avg far %.3f, num close %d num far %d',
                avg_dist_close, avg_dist_far, num_close, num_far,
            )

            # noinspection PyShadowingBuiltins
            bin = 1

            dist_limit = 4000
            dist_bins_close = np.zeros(dist_limit + 1, dtype=np.int32)
            dist_bins_far = np.zeros(dist_limit + 1, dtype=np.int32)

            max_time = 5000
            first_bins_close = np.zeros(max_time + 1, dtype=np.int32)
            first_bins_far = np.zeros(max_time + 1, dtype=np.int32)
            second_bins_close = np.zeros(max_time + 1, dtype=np.int32)
            second_bins_far = np.zeros(max_time + 1, dtype=np.int32)

            if not self.close_buff.empty() and not self.far_buff.empty():
                dist = self.close_buff.dist
                idx_first, idx_second = self.close_buff.idx_first, self.close_buff.idx_second
                for i in range(len(self.close_buff)):
                    dist_bins_close[dist[i] // bin] += 1
                    first_bins_close[idx_first[i] // bin] += 1
                    second_bins_close[idx_second[i] // bin] += 1

                dist = self.far_buff.dist
                idx_first, idx_second = self.far_buff.idx_first, self.far_buff.idx_second
                for i in range(len(self.far_buff)):
                    dist_bins_far[dist[i] // bin] += 1
                    first_bins_far[idx_first[i] // bin] += 1
                    second_bins_far[idx_second[i] // bin] += 1

        with timing.timeit('trajectories'):
            for trajectory in trajectories:
                if len(trajectory) <= 1:
                    continue

                obs = trajectory.obs

                num_deliberate = [int(trajectory.deliberate_action[0])] * len(trajectory)
                deliberate_indices = []
                if num_deliberate[0]:
                    deliberate_indices = [0]

                for i in range(1, len(trajectory)):
                    num_deliberate[i] = num_deliberate[i - 1]
                    if trajectory.deliberate_action[i]:
                        num_deliberate[i] += 1
                        deliberate_indices.append(i)

                assert len(deliberate_indices) <= len(trajectory)

                curr_j, close_j, far_j, limit_j = 0, 0, 0, 0
                for i in range(len(trajectory)):
                    # closest deliberate action to the current observation
                    while curr_j < len(deliberate_indices):
                        if deliberate_indices[curr_j] >= i:
                            break
                        curr_j += 1

                    while close_j < len(deliberate_indices):
                        if num_deliberate[deliberate_indices[close_j]] - num_deliberate[i] >= close:
                            break
                        close_j += 1

                    while far_j < len(deliberate_indices):
                        if num_deliberate[deliberate_indices[far_j]] - num_deliberate[i] >= far:
                            break
                        far_j += 1

                    while limit_j < len(deliberate_indices):
                        if deliberate_indices[limit_j] - i > dist_limit:
                            break
                        limit_j += 1

                    assert far_j >= close_j

                    # for each frame sample one "close" and one "far" example
                    close_indices = deliberate_indices[curr_j:min(close_j, limit_j)]
                    far_indices = deliberate_indices[far_j:limit_j]

                    if len(close_indices) > 0:
                        # first: sample "close" example
                        second_idx = random.choice(close_indices)
                        dist = second_idx - i

                        balanced_dist = dist_bins_close[dist // bin] <= 1.2 * dist_bins_far[dist // bin] + 5
                        balanced_first = first_bins_close[i // bin] <= 2 * first_bins_far[i // bin] + 5
                        balanced_second = second_bins_close[second_idx // bin] <= 1.2 * second_bins_far[second_idx // bin] + 5
                        balanced = balanced_dist and balanced_first and balanced_second

                        if balanced:
                            close_buff.add(
                                obs_first=obs[i],
                                obs_second=obs[second_idx],
                                labels=0,
                                dist=dist,
                                idx_first=i,
                                idx_second=second_idx,
                            )

                            sum_dist_close += dist
                            num_close += 1
                            dist_bins_close[dist // bin] += 1
                            first_bins_close[i // bin] += 1
                            second_bins_close[second_idx // bin] += 1

                    if len(far_indices) > 0:
                        # sample "far" example
                        second_idx = random.choice(far_indices)
                        dist = second_idx - i

                        balanced_dist = dist_bins_far[dist // bin] <= 1.2 * dist_bins_close[dist // bin] + 1
                        balanced_first = first_bins_far[i // bin] <= 1.2 * first_bins_close[i // bin] + 1
                        balanced_second = second_bins_far[second_idx // bin] <= 1.2 * second_bins_close[second_idx // bin] + 1
                        balanced = balanced_dist and balanced_first and balanced_second

                        if balanced:
                            far_buff.add(
                                obs_first=obs[i],
                                obs_second=obs[second_idx],
                                labels=1,
                                dist=dist,
                                idx_first=i,
                                idx_second=second_idx,
                            )
                            sum_dist_far += dist
                            num_far += 1
                            dist_bins_far[dist // bin] += 1
                            first_bins_far[i // bin] += 1
                            second_bins_far[second_idx // bin] += 1

        log.info('Close examples %d far examples %d', len(close_buff), len(far_buff))

        with timing.timeit('add_batch'):
            close_buff.shuffle_data()
            far_buff.shuffle_data()
            self.close_buff.add_buff(close_buff)
            self.far_buff.add_buff(far_buff)

            for buff in [self.close_buff, self.far_buff]:
                buff.shuffle_data()
                buff.trim_at(self.params.reachability_target_buffer_size // 2)

        with timing.timeit('finalize'):
            num_examples = min(len(self.close_buff), len(self.far_buff))
            max_close, max_far = num_examples, num_examples

            self.buffer.clear()
            with timing.timeit('add_buffers'):
                self.buffer.add_buff(self.close_buff, max_to_add=max_close)
                self.buffer.add_buff(self.far_buff, max_to_add=max_far)
                self.shuffle_data()

        if self.batch_num % 6 == 0:
            with timing.timeit('visualize'):
                self._visualize_data()

        self.batch_num += 1
        log.info('Reachability timing %s', timing)

    def has_enough_data(self):
        len_data, min_data = len(self.buffer), self.params.reachability_target_buffer_size // 20
        if len_data < min_data:
            log.info('Need to gather more data to train reachability net, %d/%d', len_data, min_data)
            return False
        return True

    def shuffle_data(self):
        self.buffer.shuffle_data()

    @staticmethod
    def _gen_histogram(folder, name, buff, close_indices, far_indices):
        bins = np.arange(np.round(buff.min()) - 1, np.round(buff.max()) + 1, dtype=np.float32, step=3)
        plt.hist(buff[close_indices], alpha=0.5, bins=bins, label='close')
        plt.hist(buff[far_indices], alpha=0.5, bins=bins, label='far')
        plt.legend(loc='upper right')
        plt.savefig(join(folder, name))
        figure = plt.gcf()
        figure.clear()

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

            if len(close_examples) >= min_vis and len(far_examples) >= min_vis:
                break

        if len(close_examples) < min_vis or len(far_examples) < min_vis:
            return

        img_folder = vis_dir(self.params.experiment_dir())
        img_folder = join(img_folder, f'reach_{time.time()}')
        ensure_dir_exists(img_folder)

        close_indices = np.nonzero(labels == 0)[0]
        far_indices = np.nonzero(labels == 1)[0]

        self._gen_histogram(img_folder, 'hist_dist.png', self.buffer.dist, close_indices, far_indices)
        self._gen_histogram(img_folder, 'hist_first_idx.png', self.buffer.idx_first, close_indices, far_indices)
        self._gen_histogram(img_folder, 'hist_second_idx.png', self.buffer.idx_second, close_indices, far_indices)

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
