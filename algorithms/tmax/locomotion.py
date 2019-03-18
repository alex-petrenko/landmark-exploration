import os
import random
import shutil
import time
from collections import deque
from os.path import join

import tensorflow as tf

from algorithms.buffer import Buffer
from algorithms.encoders import make_encoder
from algorithms.env_wrappers import main_observation_space
from algorithms.tf_utils import placeholders_from_spaces, placeholder_from_space, dense
from algorithms.tmax.tmax_utils import TmaxMode
from utils.distributions import CategoricalProbabilityDistribution
from utils.gifs import encode_gif
from utils.utils import log, vis_dir, ensure_dir_exists


class LocomotionNetwork:
    def __init__(self, env, params):
        obs_space = main_observation_space(env)
        self.ph_obs_curr, self.ph_obs_goal = placeholders_from_spaces(obs_space, obs_space)
        self.ph_actions = placeholder_from_space(env.action_space)

        with tf.variable_scope('loco'):
            encoder = tf.make_template(
                'siamese_enc_loco', make_encoder, create_scope_now_=True, env=env, regularizer=None, params=params,
            )

            obs_curr_encoded = encoder(self.ph_obs_curr).encoded_input
            obs_goal_encoded = encoder(self.ph_obs_goal).encoded_input
            obs_encoded = tf.concat([obs_curr_encoded, obs_goal_encoded], axis=1)

            fc_layers = [256, 256]
            x = obs_encoded
            for fc_layer_size in fc_layers:
                x = dense(x, fc_layer_size)

            action_logits = tf.layers.dense(x, env.action_space.n, activation=None)
            self.actions_distribution = CategoricalProbabilityDistribution(action_logits)
            self.best_action_deterministic = tf.argmax(action_logits, axis=1)
            self.act = self.actions_distribution.sample()

            actions_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ph_actions, logits=action_logits)
            self.actions_loss = tf.reduce_mean(actions_loss)

            # self.loss = self.actions_loss + self.distance_loss
            self.loss = self.actions_loss

    def navigate(self, session, obs_curr, obs_goal, deterministic=False):
        actions = session.run(
            self.best_action_deterministic if deterministic else self.act,
            feed_dict={self.ph_obs_curr: obs_curr, self.ph_obs_goal: obs_goal},
        )
        return actions


class LocomotionBuffer:
    """
    Training data for the locomotion network. Pairs of (current observation, goal observation) and "ground truth"
    actions taken from the "successful" trajectory (that managed to reach the goal).

    This is very similar to reachability buffer, there's a potential for refactoring to reuse some code.
    """

    def __init__(self, params):
        self.params = params
        self.buffer = Buffer()
        self._vis_dirs = deque([])

    def extract_data(self, trajectories):
        training_data = []

        for trajectory in trajectories:
            landmarks = []  # indices of "key" observations
            for i in range(len(trajectory)):
                if trajectory.is_landmark[i]:
                    landmarks.append(i)
            assert len(landmarks) > 0

            if len(landmarks) <= 1:
                continue

            for i in range(1, len(landmarks)):
                l_prev, l_next = landmarks[i - 1], landmarks[i]

                deliberate_actions = sum(trajectory.deliberate_action[l_prev:l_next]) == l_next - l_prev
                if not deliberate_actions:
                    # don't train locomotion on "idle" actions
                    continue

                assert l_next > l_prev
                traj_len = l_next - l_prev
                if traj_len > self.params.locomotion_max_trajectory:
                    # trajectory is too long and probably too noisy
                    # log.info('Trajectory too long %d', traj_len)
                    continue

                traj_buffer = Buffer()

                for j in range(l_prev, l_next):
                    traj_buffer.add(
                        obs_curr=trajectory.obs[j],
                        obs_goal=trajectory.obs[l_next],
                        actions=trajectory.actions[j],
                        mode=trajectory.modes[j],
                    )

                training_data.append(traj_buffer)

            if sum((len(buff) for buff in training_data)) > self.params.locomotion_target_buffer_size // 5:
                break

        if len(training_data) <= 0:
            # no new data
            return

        self._visualize_data(training_data)

        for traj_buffer in training_data:
            self.buffer.add_buff(traj_buffer)
            self.shuffle_data()
            self.buffer.trim_at(self.params.locomotion_target_buffer_size)

    def has_enough_data(self):
        len_data, min_data = len(self.buffer), self.params.locomotion_target_buffer_size // 20
        if len_data < min_data:
            log.info('Need to gather more data to train locomotion net, %d/%d', len_data, min_data)
            return False
        return True

    def shuffle_data(self):
        self.buffer.shuffle_data()

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
