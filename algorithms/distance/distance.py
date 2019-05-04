import os
import random
import shutil
import time
from collections import deque
from functools import partial
from os.path import join

import cv2
import numpy as np
import tensorflow as tf

from algorithms.tmax.tmax_utils import TmaxTrajectory, TmaxMode
from algorithms.topological_maps.topological_map import hash_observation
from algorithms.utils.buffer import Buffer
from algorithms.utils.encoders import make_encoder, EncoderParams
from algorithms.utils.env_wrappers import main_observation_space
from algorithms.utils.observation_encoder import ObservationEncoder
from algorithms.utils.tf_utils import dense, placeholders_from_spaces, merge_summaries
from utils.timing import Timing
from utils.utils import log, vis_dir, ensure_dir_exists


class DistanceNetworkParams:
    def __init__(self):
        self.close_threshold = 5  # num. of frames between obs, such that one is close to the other
        self.far_threshold = 25  # num. of frames between obs, such that one is far from the other
        self.distance_target_buffer_size = 140000  # target number of training examples to store
        self.distance_train_epochs = 8
        self.distance_batch_size = 128
        self.distance_bootstrap = 4000000
        self.distance_train_interval = 1000000
        self.distance_symmetric = True  # useful in 3D environments like Doom and DMLab

        self.distance_encoder = 'convnet_84px'
        self.distance_use_batch_norm = False
        self.distance_fc_num = 2
        self.distance_fc_size = 256


class DistanceNetwork:
    def __init__(self, env, params):
        obs_space = main_observation_space(env)
        self.ph_obs_first, self.ph_obs_second = placeholders_from_spaces(obs_space, obs_space)
        self.ph_labels = tf.placeholder(dtype=tf.int32, shape=(None,))
        self.ph_is_training = tf.placeholder(dtype=tf.bool, shape=[])

        with tf.variable_scope('distance') as scope:
            self.step = tf.Variable(0, trainable=False, dtype=tf.int64, name='dist_step')
            reg = tf.contrib.layers.l2_regularizer(scale=1e-5)
            summary_collections = ['dist']

            enc_params = EncoderParams()
            enc_params.enc_name = params.distance_encoder
            enc_params.batch_norm = params.distance_use_batch_norm
            enc_params.ph_is_training = self.ph_is_training
            enc_params.summary_collections = summary_collections

            encoder = tf.make_template(
                'siamese_enc', make_encoder, create_scope_now_=True,
                obs_space=obs_space, regularizer=reg, enc_params=enc_params,
            )

            obs_first_enc = encoder(self.ph_obs_first)
            obs_second_enc = encoder(self.ph_obs_second)

            self.first_encoded = obs_first_enc.encoded_input
            self.second_encoded = obs_second_enc.encoded_input

            observations_encoded = tf.concat([self.first_encoded, self.second_encoded], axis=1)

            fc_layers = [params.distance_fc_size] * params.distance_fc_num
            x = observations_encoded
            for fc_layer_size in fc_layers:
                x = dense(
                    x, fc_layer_size, reg, batch_norm=params.distance_use_batch_norm, is_training=self.ph_is_training,
                )

            logits = tf.contrib.layers.fully_connected(x, 2, activation_fn=None)
            self.probabilities = tf.nn.softmax(logits)
            self.correct = tf.reduce_mean(
                tf.to_float(tf.equal(self.ph_labels, tf.cast(tf.argmax(logits, axis=1), tf.int32))),
            )

            self.dist_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.ph_labels)
            self.dist_loss = tf.reduce_mean(self.dist_loss)

            reg_losses = tf.losses.get_regularization_losses(scope=scope.name)
            self.reg_loss = tf.reduce_sum(reg_losses)

            self.loss = self.dist_loss + self.reg_loss

            # helpers to encode observations (saves time)
            # does not matter if we use first vs second here
            self.ph_obs = self.ph_obs_first
            self.encoded_observation = self.first_encoded

            self._add_summaries(summary_collections)
            self.summaries = merge_summaries(collections=summary_collections)

            opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate, name='dist_opt')
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                self.train_op = opt.minimize(self.loss, global_step=self.step)

        # other stuff not related to computation graph
        self.obs_encoder = ObservationEncoder(encode_func=self.encode_observation)

    def _add_summaries(self, collections):
        with tf.name_scope('distance'):
            distance_scalar = partial(tf.summary.scalar, collections=collections)
            distance_scalar('dist_steps', self.step)
            distance_scalar('dist_loss', self.dist_loss)
            distance_scalar('dist_correct', self.correct)
            distance_scalar('dist_reg_loss', self.reg_loss)

    def get_probabilities(self, session, obs_first_encoded, obs_second_encoded):
        assert len(obs_first_encoded) == len(obs_second_encoded)
        if len(obs_first_encoded) <= 0:
            return []

        probabilities = session.run(
            self.probabilities,
            feed_dict={
                self.first_encoded: obs_first_encoded,
                self.second_encoded: obs_second_encoded,
                self.ph_is_training: False,
            },
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
        return session.run(
            self.encoded_observation, feed_dict={self.ph_obs: obs, self.ph_is_training: False},
        )

    def train(self, buffer, env_steps, agent, timing=None):
        if timing is None:
            timing = Timing()

        params = agent.params

        batch_size = params.distance_batch_size
        summary = None
        dist_step = self.step.eval(session=agent.session)

        prev_loss = 1e10
        num_epochs = params.distance_train_epochs

        log.info('Train distance net %d pairs, batch %d, epochs %d', len(buffer), batch_size, num_epochs)

        with timing.timeit('dist_epochs'):
            for epoch in range(num_epochs):
                losses = []

                with timing.add_time('shuffle'):
                    buffer.shuffle_data()

                obs_first, obs_second, labels = buffer.obs_first, buffer.obs_second, buffer.labels

                with timing.add_time('batch'):
                    for i in range(0, len(obs_first) - 1, batch_size):
                        # noinspection PyProtectedMember
                        with_summaries = agent._should_write_summaries(dist_step) and summary is None
                        summaries = [self.summaries] if with_summaries else []

                        start, end = i, i + batch_size

                        result = agent.session.run(
                            [self.loss, self.train_op] + summaries,
                            feed_dict={
                                self.ph_obs_first: obs_first[start:end],
                                self.ph_obs_second: obs_second[start:end],
                                self.ph_labels: labels[start:end],
                                self.ph_is_training: True,
                            }
                        )

                        dist_step += 1
                        # noinspection PyProtectedMember
                        agent._maybe_save(dist_step, env_steps)
                        losses.append(result[0])

                        if with_summaries:
                            summary = result[-1]
                            agent.summary_writer.add_summary(summary, global_step=env_steps)

                    # check loss improvement at the end of each epoch, early stop if necessary
                    avg_loss = np.mean(losses)
                    if avg_loss >= prev_loss:
                        log.info('Early stopping after %d epochs because distance net did not improve', epoch + 1)
                        log.info('Was %.4f now %.4f, ratio %.3f', prev_loss, avg_loss, avg_loss / prev_loss)
                        break
                    prev_loss = avg_loss

        return dist_step


class DistanceNetworkResnet:
    # noinspection PyUnusedLocal
    def __init__(self, env, params):
        from keras import Input, Model
        from keras.layers import Lambda, concatenate

        # noinspection PyProtectedMember
        from algorithms.architectures.resnet_keras import ResnetBuilder, _top_network

        width, height, channels = main_observation_space(env)
        size_embedding = 512

        with tf.variable_scope('distance'):
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


class DistanceBuffer:
    """Training data for the distance network (observation pairs and labels)."""

    def __init__(self, params):
        self.buffer = Buffer()
        self.close_buff, self.far_buff = Buffer(), Buffer()
        self.batch_num = 0

        self._vis_dirs = deque([])

        self.num_trajectories_to_process = 20
        self.complete_trajectories = deque([])

        self.params = params

    def extract_data(self, trajectories):
        timing = Timing()

        if len(self.buffer) > self.params.distance_target_buffer_size:
            # already enough data
            return

        close, far = self.params.close_threshold, self.params.far_threshold

        num_close, num_far = 0, 0
        data_added = 0

        with timing.timeit('trajectories'):
            for trajectory in trajectories:
                check_tmax = isinstance(trajectory, TmaxTrajectory)

                obs = trajectory.obs

                indices = list(range(len(trajectory)))
                np.random.shuffle(indices)

                for i in indices:
                    if len(self.buffer) > self.params.distance_target_buffer_size // 2:
                        if data_added > self.params.distance_target_buffer_size // 4:  # to limit memory usage
                            break

                    close_i = min(i + close, len(trajectory))
                    far_i = min(i + far, len(trajectory))

                    # sample close observation pair
                    first_idx = i
                    second_idx = np.random.randint(i, close_i)

                    # in TMAX we do some additional checks
                    add_close = True
                    if check_tmax:
                        both_frames_random = trajectory.is_random[first_idx] and trajectory.is_random[second_idx]
                        first_exploration = trajectory.mode[first_idx] == TmaxMode.EXPLORATION
                        second_exploration = trajectory.mode[second_idx] == TmaxMode.EXPLORATION
                        if both_frames_random or (first_exploration and second_exploration):
                            add_close = True
                        else:
                            add_close = False

                    if add_close:
                        if self.params.distance_symmetric and random.random() < 0.5:
                            first_idx, second_idx = second_idx, first_idx

                        self.buffer.add(obs_first=obs[first_idx], obs_second=obs[second_idx], labels=0)
                        data_added += 1
                        num_close += 1

                    # sample far observation pair
                    if far_i < len(trajectory):
                        first_idx = i
                        second_idx = np.random.randint(far_i, len(trajectory))

                        add_far = True
                        if check_tmax:
                            both_frames_random = trajectory.is_random[first_idx] and trajectory.is_random[second_idx]
                            first_exploration = trajectory.mode[first_idx] == TmaxMode.EXPLORATION
                            second_exploration = trajectory.mode[second_idx] == TmaxMode.EXPLORATION
                            if both_frames_random or (first_exploration and second_exploration):
                                add_far = True
                            else:
                                add_far = False

                        if add_far:
                            if self.params.distance_symmetric and random.random() < 0.5:
                                first_idx, second_idx = second_idx, first_idx

                            self.buffer.add(obs_first=obs[first_idx], obs_second=obs[second_idx], labels=1)
                            data_added += 1
                            num_far += 1

        with timing.timeit('finalize'):
            self.buffer.trim_at(self.params.distance_target_buffer_size)

        if self.batch_num % 20 == 0:
            with timing.timeit('visualize'):
                self._visualize_data()

        self.batch_num += 1
        log.info('num close %d, num far %d, distance net timing %s', num_close, num_far, timing)

    def has_enough_data(self):
        len_data, min_data = len(self.buffer), self.params.distance_target_buffer_size // 3
        if len_data < min_data:
            log.info('Need to gather more data to train distance net, %d/%d', len_data, min_data)
            return False
        return True

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
        img_folder = ensure_dir_exists(join(img_folder, 'dist'))
        img_folder = ensure_dir_exists(join(img_folder, f'dist_{time.time()}'))

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
