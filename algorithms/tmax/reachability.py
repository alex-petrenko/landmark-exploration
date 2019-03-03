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

            self.reg = tf.contrib.layers.l2_regularizer(scale=1e-5)

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

            self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            self.loss = self.reach_loss + self.first_loss + self.second_loss + self.reg_loss

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
        class Frame:
            def __init__(self, idx_):
                self.i = idx_
                self.close, self.far = False, False
                self.fail_close, self.fail_far = False, False
                self.close_i = self.far_i = -1

            def set_limits(self, close_i_, far_i_):
                self.close_i = close_i_
                self.far_i = far_i_

        close = self.params.reachable_threshold
        far = self.params.unreachable_threshold
        far_limit = 200
        close_in_time = far

        timing = Timing()
        data = Buffer()

        with timing.timeit('trajectories'):
            for trajectory in trajectories:
                if len(trajectory) <= 1:
                    continue

                first_frames = [Frame(i) for i in range(len(trajectory))]
                second_frames = [Frame(i) for i in range(len(trajectory))]

                indices = np.arange(len(trajectory))
                np.random.shuffle(indices)

                obs = trajectory.obs

                num_deliberate = [int(trajectory.deliberate_action[0])] * len(trajectory)
                for i in range(1, len(trajectory)):
                    num_deliberate[i] = num_deliberate[i - 1]
                    if trajectory.deliberate_action[i]:
                        num_deliberate[i] += 1

                close_i = far_i = 0
                for i in range(len(trajectory)):
                    while close_i < len(trajectory) and num_deliberate[close_i] - num_deliberate[i] <= close:
                        close_i += 1
                    while far_i < len(trajectory) and num_deliberate[far_i] - num_deliberate[i] <= far:
                        far_i += 1
                    first_frames[i].set_limits(close_i, far_i)

                close_i = far_i = len(trajectory) - 1
                for i in range(len(trajectory) - 1, -1, -1):
                    while close_i >= 0 and num_deliberate[i] - num_deliberate[close_i] <= close:
                        close_i -= 1
                    while far_i >= 0 and num_deliberate[i] - num_deliberate[far_i] <= far:
                        far_i -= 1
                    second_frames[i].set_limits(close_i, far_i)

                for idx in indices:
                    frame = first_frames[idx]
                    if frame.far and frame.close:
                        continue
                    if frame.fail_close or frame.fail_far:
                        continue

                    q = deque([])
                    q.append((frame, True))

                    buffer = Buffer()

                    while len(q) > 0:
                        frame, is_first = q.popleft()

                        if is_first:
                            second_frame_close, second_frame_far = None, None

                            if not frame.close:
                                # trying to find second close observation
                                second_close_i = -1
                                close_indices = list(range(frame.i + 1, frame.close_i))
                                random.shuffle(close_indices)

                                for i in close_indices:
                                    if second_frames[i].close:
                                        continue

                                    second_close_i = i
                                    break

                                if second_close_i == -1:
                                    # failed to find "close" observation, skip
                                    frame.fail_close = True
                                    continue

                                second_frame_close = second_frames[second_close_i]

                            if not frame.far:
                                # trying to find second far observation
                                second_far_i = -1
                                far_indices = list(range(frame.far_i, min(len(trajectory), frame.far_i + far_limit)))
                                random.shuffle(far_indices)

                                for i in far_indices:
                                    if second_frames[i].far:
                                        continue

                                    second_far_i = i
                                    break

                                if second_far_i == -1:
                                    # failed to find "far" observation, skip
                                    frame.fail_far = True
                                    continue

                                second_frame_far = second_frames[second_far_i]

                            if second_frame_close is not None:
                                frame.close = second_frame_close.close = True
                                q.append((second_frame_close, False))
                                buffer.add(idx_first=frame.i, idx_second=second_frame_close.i, label=0)
                            if second_frame_far is not None:
                                frame.far = second_frame_far.far = True
                                q.append((second_frame_far, False))
                                buffer.add(idx_first=frame.i, idx_second=second_frame_far.i, label=1)

                        else:  # this is a second observation in a pair
                            first_frame_close, first_frame_far = None, None

                            if not frame.close:
                                # trying to find first close observation
                                first_close_i = -1
                                close_indices = list(range(frame.i - 1, frame.close_i, -1))
                                random.shuffle(close_indices)

                                for i in close_indices:
                                    if first_frames[i].close:
                                        continue

                                    first_close_i = i
                                    break

                                if first_close_i == -1:
                                    # failed to find "close" observation, skip
                                    frame.fail_close = True
                                    continue

                                first_frame_close = first_frames[first_close_i]

                            if not frame.far:
                                # trying to find first far observation
                                first_far_i = -1
                                far_indices = list(range(max(0, frame.far_i - far_limit), frame.far_i))
                                random.shuffle(far_indices)

                                for i in far_indices:
                                    if first_frames[i].far:
                                        continue

                                    first_far_i = i
                                    break

                                if first_far_i == -1:
                                    # failed to find "far" observation, skip
                                    frame.fail_far = True
                                    continue

                                first_frame_far = first_frames[first_far_i]

                            if first_frame_close is not None:
                                frame.close = first_frame_close.close = True
                                q.append((first_frame_close, True))
                                buffer.add(idx_first=first_frame_close.i, idx_second=frame.i, label=0)
                            if first_frame_far is not None:
                                frame.far = first_frame_far.far = True
                                q.append((first_frame_far, False))
                                buffer.add(idx_first=first_frame_far.i, idx_second=frame.i, label=1)

                    # end while loop
                    if len(buffer) >= 5:
                        log.info('Buffer size %d,  trajectory len %d', len(buffer), len(trajectory))
                        idx_first = buffer.idx_first
                        idx_second = buffer.idx_second
                        labels = buffer.label

                        num_far_in_time = 0
                        for i in range(len(buffer)):
                            dist = idx_second[i] - idx_first[i]
                            if labels[i] == 0 and dist >= close_in_time:
                                num_far_in_time += 1

                        if num_far_in_time <= 0:
                            log.info('Buffer does not contain far_in_time close observations')
                        else:
                            for i in range(len(buffer)):
                                i1, i2 = idx_first[i], idx_second[i]
                                dist = i2 - i1
                                label = labels[i]

                                data.add(
                                    obs_first=obs[idx_first[i]],
                                    obs_second=obs[idx_second[i]],
                                    labels=label,
                                    dist=dist,
                                    idx_first=i1,
                                    idx_second=i2,
                                )

        with timing.timeit('add_and_shuffle'):
            self.buffer.add_buff(data)
            self.shuffle_data()
            self.buffer.trim_at(self.params.reachability_target_buffer_size)

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
        bins = np.arange(np.round(buff.min()) - 1, np.round(buff.max()) + 1, dtype=np.float32)
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
