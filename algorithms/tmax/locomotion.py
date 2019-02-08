import numpy as np

import tensorflow as tf

from algorithms.encoders import make_encoder
from algorithms.env_wrappers import get_observation_space
from algorithms.tf_utils import placeholders_from_spaces, placeholder_from_space, dense
from utils.distributions import CategoricalProbabilityDistribution
from utils.utils import log


class LocomotionNetwork:
    def __init__(self, env, params):
        obs_space = get_observation_space(env)
        self.ph_obs_curr, self.ph_obs_goal = placeholders_from_spaces(obs_space, obs_space)
        self.ph_actions = placeholder_from_space(env.action_space)

        with tf.variable_scope('locomotion'):
            encoder = tf.make_template(
                'siamese_enc_loc', make_encoder, create_scope_now_=True, env=env, regularizer=None, params=params,
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
            self.act = self.actions_distribution.sample()

            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.ph_actions, logits=action_logits,
            ))

    def navigate(self, session, obs_curr, obs_goal):
        actions = session.run(self.act, feed_dict={self.ph_obs_curr: obs_curr, self.ph_obs_goal: obs_goal})
        return actions


class LocomotionBuffer:
    """
    Training data for the locomotion network. Pairs of (current observation, goal observation) and "ground truth"
    actions taken from the "successful" trajectory (that managed to reach the goal).

    This is very similar to reachability buffer, there's a potential for refactoring to reuse some code.
    """

    def __init__(self, params):
        self.obs_curr, self.obs_goal, self.actions = [], [], []
        self.params = params

    def extract_data(self, trajectories):
        obs_curr, obs_goal, actions = [], [], []

        for trajectory in trajectories:
            landmarks = trajectory.landmarks  # indices of "key" observations
            if len(landmarks) <= 1:
                continue

            for i in range(1, len(landmarks)):
                l_prev, l_next = landmarks[i - 1], landmarks[i]
                assert l_next > l_prev
                traj_len = l_next - l_prev
                if traj_len > self.params.locomotion_max_trajectory:
                    # trajectory is too long and probably too noisy
                    continue

                for j in range(l_prev, l_next):
                    obs_curr.append(trajectory.obs[j])
                    obs_goal.append(trajectory.obs[l_next])
                    actions.append(trajectory.actions[j])

        if len(obs_curr) <= 0:
            # no new data
            return

        if len(self.obs_curr) <= 0:
            self.obs_curr = np.array(obs_curr)
            self.obs_goal = np.array(obs_goal)
            self.actions = np.array(actions, dtype=np.int32)
        else:
            self.obs_curr = np.append(self.obs_curr, obs_curr, axis=0)
            self.obs_goal = np.append(self.obs_goal, obs_goal, axis=0)
            self.actions = np.append(self.actions, actions, axis=0)

        self._discard_data()

        assert len(self.obs_curr) == len(self.obs_goal)
        assert len(self.obs_curr) == len(self.actions)

    def _discard_data(self):
        """Remove some data if the current buffer is too big."""
        target_size = self.params.locomotion_target_buffer_size
        if len(self.obs_curr) <= target_size:
            return

        self.shuffle_data()
        self.obs_curr = self.obs_curr[:target_size]
        self.obs_goal = self.obs_goal[:target_size]
        self.actions = self.actions[:target_size]

    def has_enough_data(self):
        len_data, min_data = len(self.obs_curr), self.params.locomotion_target_buffer_size // 10
        if len_data < min_data:
            log.info('Need to gather more data to train locomotion net, %d/%d', len_data, min_data)
            return False
        return True

    def shuffle_data(self):
        if len(self.obs_curr) <= 0:
            return

        chaos = np.random.permutation(len(self.obs_curr))
        self.obs_curr = self.obs_curr[chaos]
        self.obs_goal = self.obs_goal[chaos]
        self.actions = self.actions[chaos]
