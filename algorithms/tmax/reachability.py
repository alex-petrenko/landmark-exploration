import os
import random
import shutil
import time
from collections import deque
from os.path import join

import cv2
import numpy as np
import tensorflow as tf

from algorithms.encoders import make_encoder
from algorithms.env_wrappers import get_observation_space
from algorithms.tf_utils import dense, placeholders_from_spaces
from algorithms.tmax.tmax_utils import TmaxMode
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
        self.obs_first, self.obs_second = [], []
        self.labels, self.dist = np.array([]), np.array([])

        self._vis_dirs = deque([])

        self.params = params

    def extract_data(self, trajectories):
        close = self.params.reachable_threshold
        far = self.params.unreachable_threshold

        obs_first_close, obs_second_close, labels_close = [], [], []
        obs_first_far, obs_second_far, labels_far = [], [], []
        dist_close, dist_far = [], []

        avg_dist_close, avg_dist_far = 0, 0

        close_indices = np.nonzero(self.labels == 0)[0]
        far_indices = np.nonzero(self.labels == 1)[0]

        sum_dist_close, sum_dist_far = np.sum(self.dist[close_indices]), np.sum(self.dist[far_indices])
        num_close, num_far = len(close_indices), len(far_indices)
        assert num_close + num_far == len(self.labels)

        if num_close > 0:
            avg_dist_close = sum_dist_close / num_close
        if num_far > 0:
            avg_dist_far = sum_dist_far / num_far
        log.info('Avg close %.3f avg far %.3f', avg_dist_close, avg_dist_far)

        for trajectory in trajectories:
            if len(trajectory) <= 1:
                continue

            if trajectory.modes[-1] != TmaxMode.EXPLORATION:
                # the entire trajectory is not exploration
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

            if len(deliberate_indices) == len(trajectory):
                # trajectory does not contain an idle segment
                continue

            assert len(deliberate_indices) < len(trajectory)

            curr_j, close_j, far_j = 0, 0, 0
            for i in range(len(trajectory)):
                if trajectory.modes[i] != TmaxMode.EXPLORATION:
                    continue

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

                assert far_j >= close_j

                # for each frame sample one "close" and one "far" example
                close_indices = deliberate_indices[curr_j:close_j]
                far_indices = deliberate_indices[far_j:]

                # don't have anywhere to sample from
                if len(close_indices) <= 0 or len(far_indices) <= 10:
                    continue

                if len(close_indices) > 0:
                    # first: sample "close" example
                    second_idx = random.choice(close_indices)
                    dist = second_idx - i

                    if num_deliberate[second_idx] - num_deliberate[i] == dist:
                        # we don't want trajectories only with deliberate actions
                        pass
                    else:
                        if num_close > 0:
                            avg_dist_close = sum_dist_close / num_close
                        if num_far > 0:
                            avg_dist_far = sum_dist_far / num_far

                        if avg_dist_close <= avg_dist_far and dist < avg_dist_far:
                            # can't add
                            pass
                        else:
                            # log.info('CLOSE: avg_dist_close %.3f avg_dist_far %.3f dist %.3f', avg_dist_close, avg_dist_far, dist)
                            obs_first_close.append(obs[i])
                            obs_second_close.append(obs[second_idx])
                            labels_close.append(0)
                            dist_close.append(dist)
                            sum_dist_close += dist
                            num_close += 1

                if len(far_indices) > 0:
                    # sample "far" example
                    second_idx = random.choice(far_indices)
                    dist = second_idx - i

                    if num_deliberate[second_idx] - num_deliberate[i] == dist:
                        # we don't want trajectories only with deliberate actions
                        pass
                    else:
                        if num_close > 0:
                            avg_dist_close = sum_dist_close / num_close
                        if num_far > 0:
                            avg_dist_far = sum_dist_far / num_far

                        if avg_dist_close < avg_dist_far < dist:
                            # don't increase avg dist
                            pass
                        else:
                            # log.info('FAR  : Avg_dist_close %.3f avg_dist_far %.3f dist %.3f', avg_dist_close, avg_dist_far, dist)
                            obs_first_far.append(obs[i])
                            obs_second_far.append(obs[second_idx])
                            labels_far.append(1)
                            dist_far.append(dist)
                            sum_dist_far += dist
                            num_far += 1

            # if len(deliberate_indices) != len(trajectory):
            #     log.info('Trajectory %d deliberate %d', len(trajectory), len(deliberate_indices))

        assert len(obs_first_close) == len(obs_second_close)
        assert len(obs_first_close) == len(labels_close)
        assert len(obs_first_close) == len(dist_close)
        assert len(obs_first_far) == len(obs_second_far)
        assert len(obs_first_far) == len(labels_far)
        assert len(obs_first_far) == len(dist_far)

        log.info('Close examples %d far examples %d', len(obs_first_close), len(obs_first_far))

        num_examples = min(len(obs_first_close), len(obs_first_far))
        obs_first = obs_first_close[:num_examples] + obs_first_far[:num_examples]
        obs_second = obs_second_close[:num_examples] + obs_second_far[:num_examples]
        labels = labels_close[:num_examples] + labels_far[:num_examples]
        dist = dist_close[:num_examples] + dist_far[:num_examples]

        if len(self.obs_first) <= 0:
            self.obs_first = np.array(obs_first)
            self.obs_second = np.array(obs_second)
            self.labels = np.array(labels, dtype=np.int32)
            self.dist = np.array(dist)
        elif len(obs_first) > 0:
            self.obs_first = np.concatenate([np.asarray(obs_first), self.obs_first])
            self.obs_second = np.concatenate([np.asarray(obs_second), self.obs_second])
            self.labels = np.concatenate([np.asarray(labels), self.labels])
            self.dist = np.concatenate([np.asarray(dist), self.dist])

        self._discard_data()
        self.shuffle_data()
        self._visualize_data()

        assert len(self.obs_first) == len(self.obs_second)
        assert len(self.obs_first) == len(self.labels)

    def _discard_data(self):
        """Remove some data if the current buffer is too big."""
        target_size = self.params.reachability_target_buffer_size
        if len(self.obs_first) <= target_size:
            return

        self.obs_first = self.obs_first[:target_size]
        self.obs_second = self.obs_second[:target_size]
        self.labels = self.labels[:target_size]
        self.dist = self.dist[:target_size]

    def has_enough_data(self):
        len_data, min_data = len(self.obs_first), self.params.reachability_target_buffer_size // 20
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
        self.dist = self.dist[chaos]

    def _visualize_data(self):
        min_vis = 10
        close_examples, far_examples = [], []
        for i in range(len(self.labels)):
            if self.labels[i] == 0 and len(close_examples) < min_vis:
                close_examples.append((self.obs_first[i], self.obs_second[i]))
            elif self.labels[i] == 1 and len(far_examples) < min_vis:
                far_examples.append((self.obs_first[i], self.obs_second[i]))

            if len(close_examples) >= min_vis and len(far_examples) >= min_vis:
                break

        if len(close_examples) < min_vis:
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
        while len(self._vis_dirs) > 30:
            dir_name = self._vis_dirs.popleft()
            if os.path.isdir(dir_name):
                shutil.rmtree(dir_name)
