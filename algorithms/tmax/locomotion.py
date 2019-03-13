import numpy as np
import tensorflow as tf

from algorithms.encoders import make_encoder
from algorithms.env_wrappers import get_observation_space
from algorithms.tf_utils import placeholders_from_spaces, placeholder_from_space, dense
from algorithms.tmax.tmax_utils import TmaxMode
from utils.distributions import CategoricalProbabilityDistribution
from utils.utils import log


class LocomotionNetwork:
    def __init__(self, env, params):
        obs_space = get_observation_space(env)
        self.ph_obs_curr, self.ph_obs_goal = placeholders_from_spaces(obs_space, obs_space)
        self.ph_actions = placeholder_from_space(env.action_space)
        # self.ph_distance = placeholder(None, dtype=tf.int32)
        # self.ph_train_distance = placeholder(None, dtype=tf.int32)
        # self.ph_train_actions = placeholder(None, dtype=tf.int32)

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

            # close-far probabilities
            # distance_hidden = dense(tf.stop_gradient(x), 256)
            # distance_logits = tf.layers.dense(distance_hidden, 2, activation=None)
            # self.distance_probabilities = tf.nn.softmax(distance_logits)

            # distance_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            #     logits=distance_logits, labels=self.ph_distance,
            # )

            # train_distance = tf.to_float(self.ph_train_distance)
            # num_train_samples = tf.maximum(tf.reduce_sum(train_distance), EPS)  # to prevent division by 0
            # self.distance_loss = tf.reduce_sum(distance_loss * train_distance) / num_train_samples

            # train_actions = tf.to_float(self.ph_train_actions)
            # num_action_samples = tf.maximum(tf.reduce_sum(train_actions), EPS)
            # actions_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.ph_actions, logits=action_logits)
            # self.actions_loss = tf.reduce_sum(actions_loss * train_actions) / num_action_samples

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

    # def _get_distance_probabilities(self, session, obs_curr, obs_goal):
    #     probabilities = session.run(
    #         self.distance_probabilities,
    #         feed_dict={self.ph_obs_curr: obs_curr, self.ph_obs_goal: obs_goal},
    #     )
    #     return probabilities
    #
    # def distances(self, session, obs_curr, obs_goal):
    #     probs = self._get_distance_probabilities(session, obs_curr, obs_goal)
    #     return [p[1] for p in probs]


class LocomotionBuffer:
    """
    Training data for the locomotion network. Pairs of (current observation, goal observation) and "ground truth"
    actions taken from the "successful" trajectory (that managed to reach the goal).

    This is very similar to reachability buffer, there's a potential for refactoring to reuse some code.
    """

    def __init__(self, params):
        self.params = params

        self.visualize_trajectories = {TmaxMode.IDLE_EXPLORATION: None, TmaxMode.LOCOMOTION: None}

        self.obs_curr, self.obs_goal, self.actions = [], [], []

    def extract_data(self, trajectories):
        obs_curr, obs_goal, actions = [], [], []

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

                # locomotion_traj = all(mode == TmaxMode.LOCOMOTION for mode in trajectory.modes[l_prev:l_next])
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
                else:
                    # log.info('Trajectory okay %d', traj_len)
                    pass

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
        elif len(obs_curr) > 0:
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
        len_data, min_data = len(self.obs_curr), self.params.locomotion_target_buffer_size // 50
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
