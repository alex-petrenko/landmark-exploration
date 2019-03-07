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

from algorithms.algo_utils import EPS
from algorithms.buffer import Buffer
from algorithms.encoders import make_encoder
from algorithms.env_wrappers import get_observation_space
from algorithms.tf_utils import dense, placeholders_from_spaces
from utils.timing import Timing
from utils.utils import log, vis_dir, ensure_dir_exists


class DistanceNetwork:
    def __init__(self, env, params):
        obs_space = get_observation_space(env)
        self.ph_obs_first, self.ph_obs_second = placeholders_from_spaces(obs_space, obs_space)
        self.ph_labels = tf.placeholder(dtype=tf.int32, shape=(None, ))

        with tf.variable_scope('distance') as scope:
            reg = tf.contrib.layers.l2_regularizer(scale=1e-5)

            def make_embedding(obs):
                conv_encoder = make_encoder(obs, env, reg, params, 'dist_enc')
                x = conv_encoder.encoded_input
                x = dense(x, 256, reg)
                x = tf.contrib.layers.fully_connected(x, 32, activation_fn=None)
                return x

            embedding_net = tf.make_template('embedding_net', make_embedding, create_scope_now_=True)

            first_embedding = embedding_net(self.ph_obs_first)
            second_embedding = embedding_net(self.ph_obs_second)

            self.distances = tf.sqrt(
                tf.reduce_sum(tf.squared_difference(first_embedding, second_embedding), axis=1) + EPS,
            )
            clipped_distances = tf.clip_by_value(self.distances, 0.0, 10.0)

            distance_losses = tf.where(tf.equal(self.ph_labels, 0), self.distances, -clipped_distances)
            self.distance_loss = tf.reduce_mean(distance_losses)

            is_far = tf.to_float(self.ph_labels)
            num_close = tf.reduce_sum(1.0 - is_far) + EPS
            num_far = tf.reduce_sum(is_far) + EPS

            self.avg_loss_close = tf.reduce_sum(distance_losses * (1 - is_far)) / num_close
            self.avg_loss_far = tf.reduce_sum(distance_losses * is_far) / num_far

            self.avg_dist_close = tf.reduce_sum(self.distances * (1 - is_far)) / num_close
            self.avg_dist_far = tf.reduce_sum(self.distances * is_far) / num_far

            reg_losses = tf.losses.get_regularization_losses(scope=scope.name)
            self.reg_loss = tf.reduce_sum(reg_losses)

            self.avg_coord_first = tf.reduce_mean(tf.abs(first_embedding))
            self.avg_coord_second = tf.reduce_mean(tf.abs(second_embedding))
            self.proximity_loss = self.avg_coord_first + self.avg_coord_second

            self.loss = self.distance_loss + 0.1 * self.proximity_loss + self.reg_loss

    def get_distances(self, session, obs_first, obs_second):
        distances = session.run(
            self.distances,
            feed_dict={self.ph_obs_first: obs_first, self.ph_obs_second: obs_second},
        )
        return distances


class DistanceBuffer:
    def __init__(self, params):
        self.buffer = Buffer()
        self.close_buff, self.far_buff = Buffer(), Buffer()
        self.batch_num = 0
        self.total_data = 0

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
        far_limit = max(0, int(np.random.normal(300, 100)))
        if random.random() < 0.1:
            far_limit = 1e9

        close_in_time = far

        timing = Timing()
        data = Buffer()
        close_data = Buffer()

        with timing.timeit('trajectories'):
            for trajectory in trajectories:
                if len(trajectory) <= 1:
                    continue

                first_frames = [Frame(i) for i in range(len(trajectory))]
                second_frames = [Frame(i) for i in range(len(trajectory))]

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

                indices = np.arange(len(trajectory))
                np.random.shuffle(indices)

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
                                start, end = frame.i, frame.close_i

                                if end > start:
                                    random_idx = np.random.randint(start, end)

                                    for step in range(max(random_idx - start, end - random_idx)):
                                        for delta in (-step, step):
                                            i = random_idx + delta
                                            if start < i < end and not second_frames[i].close:
                                                second_close_i = i
                                                break

                                        if second_close_i != -1:
                                            break

                                if second_close_i == -1:
                                    frame.fail_close = True
                                    continue
                                else:
                                    second_frame_close = second_frames[second_close_i]

                            if not frame.far:
                                # trying to find second far observation
                                second_far_i = -1
                                start = frame.far_i
                                end = len(trajectory)
                                if frame.i + far_limit > frame.far_i + close_in_time:
                                    end = min(end, frame.i + far_limit)

                                if end > start:
                                    random_idx = np.random.randint(start, end)

                                    for step in range(max(random_idx - start, end - random_idx)):
                                        for delta in (-step, step):
                                            i = random_idx + delta
                                            if start < i < end and not second_frames[i].far:
                                                second_far_i = i
                                                break

                                        if second_far_i != -1:
                                            break

                                if second_far_i == -1:
                                    frame.fail_far = True
                                    continue
                                else:
                                    second_frame_far = second_frames[second_far_i]

                            if second_frame_close is not None:
                                frame.close = second_frame_close.close = True
                                buffer.add(idx_first=frame.i, idx_second=second_frame_close.i, label=0)
                                q.append((second_frame_close, False))
                            if second_frame_far is not None:
                                frame.far = second_frame_far.far = True
                                buffer.add(idx_first=frame.i, idx_second=second_frame_far.i, label=1)
                                q.append((second_frame_far, False))

                        else:  # this is a second observation in a pair
                            first_frame_close, first_frame_far = None, None

                            if not frame.close:
                                # trying to find first close observation
                                first_close_i = -1
                                start, end = frame.close_i, frame.i

                                if end > start:
                                    random_idx = np.random.randint(start, end)

                                    for step in range(max(random_idx - start, end - random_idx)):
                                        for delta in (-step, step):
                                            i = random_idx + delta
                                            if start < i < end and not first_frames[i].close:
                                                first_close_i = i
                                                break

                                        if first_close_i != -1:
                                            break

                                if first_close_i == -1:
                                    frame.fail_close = True
                                    continue
                                else:
                                    first_frame_close = first_frames[first_close_i]

                            if not frame.far:
                                # trying to find first far observation
                                first_far_i = -1
                                start, end = 0, frame.far_i
                                if frame.i - far_limit < frame.far_i - close_in_time:
                                    start = max(start, frame.i - far_limit)

                                if end > start:
                                    random_idx = np.random.randint(start, end)

                                    for step in range(max(random_idx - start, end - random_idx)):
                                        for delta in (-step, step):
                                            i = random_idx + delta
                                            if start < i < end and not first_frames[i].far:
                                                first_far_i = i
                                                break

                                        if first_far_i == -1:
                                            break

                                if first_far_i == -1:
                                    frame.fail_far = True
                                    continue
                                else:
                                    first_frame_far = first_frames[first_far_i]

                            if first_frame_close is not None:
                                frame.close = first_frame_close.close = True
                                buffer.add(idx_first=first_frame_close.i, idx_second=frame.i, label=0)
                                q.append((first_frame_close, True))
                            if first_frame_far is not None:
                                frame.far = first_frame_far.far = True
                                buffer.add(idx_first=first_frame_far.i, idx_second=frame.i, label=1)
                                q.append((first_frame_far, True))

                    # end while loop
                    if len(buffer) >= 5:
                        idx_first = buffer.idx_first
                        idx_second = buffer.idx_second
                        labels = buffer.label

                        num_far_in_time = 0
                        for i in range(len(buffer)):
                            dist = idx_second[i] - idx_first[i]
                            if labels[i] == 0 and dist >= close_in_time:
                                num_far_in_time += 1

                        if num_far_in_time <= len(buffer) // 4:
                            pass
                        else:
                            for i in range(len(buffer)):
                                i1, i2 = idx_first[i], idx_second[i]
                                dist = i2 - i1
                                label = labels[i]

                                data.add(
                                    obs_first=obs[i1],
                                    obs_second=obs[i2],
                                    labels=label,
                                    dist=dist,
                                    idx_first=i1,
                                    idx_second=i2,
                                )

                # end for loop
                np.random.shuffle(indices)

                avg_data_per_batch = self.total_data / (self.batch_num + 1)

                for idx in indices:
                    if len(close_data) > avg_data_per_batch:
                        break

                    frame = first_frames[idx]
                    if frame.close_i <= frame.i + 1:
                        continue

                    close_idx = np.random.randint(frame.i + 1, frame.close_i)
                    close_data.add(
                        obs_first=obs[frame.i],
                        obs_second=obs[close_idx],
                        labels=0,
                        dist=close_idx - frame.i,
                        idx_first=frame.i,
                        idx_second=close_idx,
                    )

        with timing.timeit('add_and_shuffle'):
            self.buffer.add_buff(data)
            self.total_data += len(data)
            self.buffer.add_buff(close_data)
            self.shuffle_data()
            self.buffer.trim_at(self.params.reachability_target_buffer_size)

        if self.batch_num % 10 == 0:
            with timing.timeit('visualize'):
                self._visualize_data()

        self.batch_num += 1
        log.info('Distance timing %s', timing)

    def has_enough_data(self):
        len_data, min_data = len(self.buffer), self.params.reachability_target_buffer_size // 40
        if len_data < min_data:
            log.info('Need to gather more data to train distance net, %d/%d', len_data, min_data)
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
