import os
import random
import shutil
import time
from collections import deque
from os.path import join

import tensorflow as tf

from algorithms.tmax.tmax_utils import TmaxMode
from algorithms.utils.buffer import Buffer
from algorithms.utils.encoders import make_encoder, EncoderParams
from algorithms.utils.env_wrappers import main_observation_space
from algorithms.utils.tf_utils import placeholders_from_spaces, placeholder_from_space, dense
from utils.distributions import CategoricalProbabilityDistribution
from utils.gifs import encode_gif
from utils.timing import Timing
from utils.utils import log, vis_dir, ensure_dir_exists


class LocomotionNetworkParams:
    def __init__(self):
        self.locomotion_experience_replay_buffer = 140000
        self.locomotion_experience_replay_epochs = 10
        self.locomotion_experience_replay_batch = 64
        self.locomotion_experience_replay_max_kl = 0.05
        self.locomotion_max_trajectory = 5  # max trajectory length to be utilized during training

        self.locomotion_encoder = 'resnet'
        self.locomotion_use_batch_norm = True
        self.locomotion_fc_num = 1
        self.locomotion_fc_size = 512


class LocomotionNetwork:
    def __init__(self, env, params):
        obs_space = main_observation_space(env)
        self.ph_obs_prev, self.ph_obs_curr, self.ph_obs_goal = placeholders_from_spaces(obs_space, obs_space, obs_space)
        self.ph_actions = placeholder_from_space(env.action_space)
        self.ph_is_training = tf.placeholder(dtype=tf.bool, shape=[])

        with tf.variable_scope('loco') as scope:
            self.step = tf.Variable(0, trainable=False, dtype=tf.int64, name='loco_step')

            reg = tf.contrib.layers.l2_regularizer(scale=1e-5)

            # encoder = tf.make_template(
            #     'siamese_enc_loco', make_encoder, create_scope_now_=True,
            #     obs_space=obs_space, regularizer=None, params=params,
            # )
            #
            # obs_curr_encoded = encoder(self.ph_obs_curr).encoded_input
            # obs_goal_encoded = encoder(self.ph_obs_goal).encoded_input
            # obs_encoded = tf.concat([obs_curr_encoded, obs_goal_encoded], axis=1)

            enc_params = EncoderParams()
            enc_params.enc_name = params.locomotion_encoder
            enc_params.batch_norm = params.locomotion_use_batch_norm
            enc_params.ph_is_training = self.ph_is_training
            enc_params.summary_collections = ['loco']

            encoder = tf.make_template(
                'joined_enc_loco', make_encoder, create_scope_now_=True,
                obs_space=obs_space, regularizer=reg, enc_params=enc_params,
            )

            obs_concat = tf.concat([self.ph_obs_prev, self.ph_obs_curr, self.ph_obs_goal], axis=2)
            obs_encoder = encoder(obs_concat)
            obs_encoded = obs_encoder.encoded_input

            encoder_reg_loss = 0.0
            if hasattr(obs_encoder, 'reg_loss'):
                encoder_reg_loss = obs_encoder.reg_loss

            fc_layers = [params.locomotion_fc_size] * params.locomotion_fc_num
            x = obs_encoded
            for fc_layer_size in fc_layers:
                x = dense(
                    x, fc_layer_size, regularizer=reg,
                    batch_norm=params.locomotion_use_batch_norm, is_training=self.ph_is_training,
                )

            action_logits = tf.layers.dense(x, env.action_space.n, activation=None)
            self.actions_distribution = CategoricalProbabilityDistribution(action_logits)
            self.best_action_deterministic = tf.argmax(action_logits, axis=1)
            self.act = self.actions_distribution.sample()

            actions_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ph_actions, logits=action_logits)
            self.actions_loss = tf.reduce_mean(actions_loss)

            reg_losses = tf.losses.get_regularization_losses(scope=scope.name)
            self.reg_loss = tf.reduce_sum(reg_losses)

            self.loss = self.actions_loss + self.reg_loss + encoder_reg_loss

            loco_opt = tf.train.AdamOptimizer(learning_rate=params.learning_rate, name='loco_opt')

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='loco')):
                self.train_loco = loco_opt.minimize(self.loss, global_step=self.step)

    def navigate(self, session, obs_prev, obs_curr, obs_goal, deterministic=False):
        actions = session.run(
            self.best_action_deterministic if deterministic else self.act,
            feed_dict={
                self.ph_obs_prev: obs_prev,
                self.ph_obs_curr: obs_curr,
                self.ph_obs_goal: obs_goal,
                self.ph_is_training: False,
            },
        )
        return actions


class LocomotionBuffer:
    """
    Training data for the hindsight experience replay (for locomotion policy).
    """

    def __init__(self, params):
        self.params = params
        self.batch_num = 0
        self.buffer = Buffer()
        self._vis_dirs = deque([])

    def extract_data(self, trajectories):
        timing = Timing()

        if len(trajectories) <= 0:
            return

        if len(self.buffer) > self.params.locomotion_experience_replay_buffer:
            return

        with timing.timeit('trajectories'):
            training_data = []
            max_trajectory = self.params.locomotion_max_trajectory

            data_so_far = 0

            trajectories = [t for t in trajectories if len(t) > self.params.locomotion_max_trajectory]

            # train only on random frames
            random_frames = [[i for i, is_random in enumerate(t.is_random) if is_random] for t in trajectories]

            total_experience = sum(len(frames) for frames in random_frames)
            max_total_experience = 0.75 * total_experience  # max fraction of experience to use
            max_num_segments = int(max_total_experience / max_trajectory)

            log.info(
                '%d total experience from %d trajectories (%d segments)',
                max_total_experience, len(trajectories), max_num_segments,
            )

            attempts = 0

            while len(training_data) < max_num_segments:
                attempts += 1
                if attempts > 100 * max_num_segments:  # just in case
                    break

                trajectory_idx = random.choice(range(len(trajectories)))
                trajectory = trajectories[trajectory_idx]
                if len(random_frames[trajectory_idx]) <= 0:
                    continue

                first_random_frame = random_frames[trajectory_idx][0]

                # sample random interval in trajectory, treat the last frame as "imaginary" goal, use actions as
                # ground truth
                start_idx = random.randint(first_random_frame, len(trajectory) - 2)
                goal_idx = min(start_idx + max_trajectory, len(trajectory) - 1)
                assert start_idx < goal_idx

                if not trajectory.is_random[start_idx]:
                    continue
                if not trajectory.is_random[goal_idx]:
                    continue

                traj_buffer = Buffer()
                for i in range(start_idx, goal_idx):
                    if not trajectory.is_random[i]:
                        continue

                    traj_buffer.add(
                        obs_prev=trajectory.obs[max(0, i - 1)],
                        obs_curr=trajectory.obs[i],
                        obs_goal=trajectory.obs[goal_idx],
                        actions=trajectory.actions[i],
                        mode=trajectory.mode[i],
                    )
                    data_so_far += 1

                training_data.append(traj_buffer)

                if len(self.buffer) + data_so_far > self.params.locomotion_experience_replay_buffer * 1.5:
                    break

        if len(training_data) <= 0:
            return

        if self.batch_num % 10 == 0:
            with timing.timeit('vis'):
                self._visualize_data(training_data)

        with timing.timeit('finalize'):
            for traj_buffer in training_data:
                self.buffer.add_buff(traj_buffer)

            # self.shuffle_data()
            # self.buffer.trim_at(self.params.locomotion_experience_replay_buffer)

        self.batch_num += 1
        log.info('Locomotion, num trajectories: %d, timing: %s', len(training_data), timing)

    def has_enough_data(self):
        len_data, min_data = len(self.buffer), self.params.locomotion_experience_replay_buffer // 3
        if len_data < min_data:
            log.info('Need to gather more data to train locomotion net, %d/%d', len_data, min_data)
            return False
        return True

    def shuffle_data(self):
        permutation = self.buffer.shuffle_data(return_permutation=True)
        return permutation

    def reset(self):
        self.buffer.clear()

    def _visualize_data(self, traj_buffers):
        data_folder = vis_dir(self.params.experiment_dir())
        data_folder = ensure_dir_exists(join(data_folder, 'loco'))
        data_folder = ensure_dir_exists(join(data_folder, f'loco_{time.time()}'))

        trajectories_by_mode = {TmaxMode.EXPLORATION: [], TmaxMode.LOCOMOTION: []}
        for i, traj_buffer in enumerate(traj_buffers):
            trajectories_by_mode[traj_buffer.mode[-1]].append(i)

        num_trajectories_to_save = 2

        for mode, indices in trajectories_by_mode.items():
            random.shuffle(indices)

            for traj_idx in indices[:num_trajectories_to_save]:
                traj_buffer = traj_buffers[traj_idx]
                gif = encode_gif(traj_buffer.obs_curr, fps=12)
                gif_name = f'{TmaxMode.mode_name(mode)}_{traj_idx}.gif'

                with open(join(data_folder, gif_name), 'wb') as gif_file:
                    gif_file.write(gif)

        self._vis_dirs.append(data_folder)
        while len(self._vis_dirs) > 10:
            dir_name = self._vis_dirs.popleft()
            if os.path.isdir(dir_name):
                shutil.rmtree(dir_name)
