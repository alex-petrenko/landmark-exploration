import numpy as np
import tensorflow as tf

from algorithms.algo_utils import EPS
from algorithms.encoders import make_encoder
from algorithms.env_wrappers import get_observation_space
from algorithms.tf_utils import placeholders_from_spaces, placeholder_from_space, dense, placeholder
from algorithms.tmax.tmax_utils import TmaxMode
from algorithms.tmax.trajectory import Trajectory
from utils.distributions import CategoricalProbabilityDistribution
from utils.utils import log


class LocomotionNetwork:
    def __init__(self, env, params):
        obs_space = get_observation_space(env)
        self.ph_obs_curr, self.ph_obs_goal = placeholders_from_spaces(obs_space, obs_space)
        self.ph_actions = placeholder_from_space(env.action_space)
        self.ph_distance = placeholder(2)
        self.ph_train_distance = placeholder(None, dtype=tf.int32)

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
            distance_hidden = dense(tf.stop_gradient(x), 256)
            distance_logits = tf.layers.dense(distance_hidden, 2, activation=None)
            self.distance_probabilities = tf.nn.softmax(distance_logits)

            distance_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=distance_logits, labels=self.ph_distance)

            train_distance = tf.to_float(self.ph_train_distance)
            num_train_samples = tf.maximum(tf.reduce_sum(train_distance), EPS)  # to prevent division by 0
            self.distance_loss = tf.reduce_sum(distance_loss * train_distance) / num_train_samples

            self.actions_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.ph_actions, logits=action_logits,
            ))

            self.loss = self.actions_loss + self.distance_loss

    def navigate(self, session, obs_curr, obs_goal, deterministic=False):
        actions = session.run(
            self.best_action_deterministic if deterministic else self.act,
            feed_dict={self.ph_obs_curr: obs_curr, self.ph_obs_goal: obs_goal},
        )
        return actions

    def _get_distance_probabilities(self, session, obs_curr, obs_goal):
        probabilities = session.run(
            self.distance_probabilities,
            feed_dict={self.ph_obs_curr: obs_curr, self.ph_obs_goal: obs_goal},
        )
        return probabilities

    def distances(self, session, obs_curr, obs_goal):
        probs = self._get_distance_probabilities(session, obs_curr, obs_goal)
        return [p[1] for p in probs]


class LocomotionBuffer:
    """
    Training data for the locomotion network. Pairs of (current observation, goal observation) and "ground truth"
    actions taken from the "successful" trajectory (that managed to reach the goal).

    This is very similar to reachability buffer, there's a potential for refactoring to reuse some code.
    """

    def __init__(self, params):
        self.obs_curr, self.obs_goal, self.actions, self.distance, self.train_distance = [], [], [], [], []
        self.params = params

        self.visualize_trajectories = {TmaxMode.EXPLORATION: None, TmaxMode.LOCOMOTION: None}

    @staticmethod
    def _calc_distance(obs_idx, goal_idx, is_idle):
        dist_frames = goal_idx - obs_idx
        near, far = 3, 10

        if is_idle:
            is_far = 0.001
        else:
            if dist_frames <= near:
                is_far = 0.001
            elif dist_frames >= far:
                is_far = 0.999
            else:
                is_far = (dist_frames - near) / (far - near)  # linear interpolation

        return 1.0 - is_far, is_far

    def extract_data(self, episode_trajectories):
        # first, split trajectories by type
        trajectories = {TmaxMode.EXPLORATION: [], TmaxMode.LOCOMOTION: []}
        for tr in episode_trajectories:
            if len(tr) <= 2:
                continue

            curr_tr = Trajectory(tr.env_idx)
            curr_tr.add(tr.obs[0], tr.actions[0], tr.modes[0], tr.target_idx[0], tr.curr_landmark_idx[0])

            for i in range(1, len(tr)):
                if curr_tr.modes[0] != tr.modes[i]:
                    # start new trajectory because mode has changed
                    trajectories[curr_tr.modes[0]].append(curr_tr)
                    curr_tr = Trajectory(tr.env_idx)

                curr_tr.add(tr.obs[i], tr.actions[i], tr.modes[i], tr.target_idx[i], tr.curr_landmark_idx[i])

            if len(curr_tr) >= 1:
                trajectories[curr_tr.modes[-1]].append(curr_tr)

        for mode, traj in trajectories.items():
            if len(traj) > 0:
                self.visualize_trajectories[mode] = traj[-1]

        obs_curr, obs_goal, actions, distance, train_distance = [], [], [], [], []
        locomotion_experience = exploration_experience = 0

        for mode in self.visualize_trajectories.keys():
            self.visualize_trajectories[mode] = None

        for tr in trajectories[TmaxMode.LOCOMOTION]:
            if len(tr) <= 2:
                continue

            target_changes = []
            for i in range(1, len(tr)):
                if tr.target_idx[i] != tr.target_idx[i - 1]:
                    target_changes.append(i)
            target_changes.append(len(tr))

            i = 0
            for target_change_idx in target_changes:
                while i < target_change_idx - 1:
                    # sample random goal from the "future"
                    goal_idx = i + self.params.locomotion_max_trajectory
                    goal_idx = min(goal_idx, target_change_idx - 1)

                    for j in range(i, goal_idx):
                        obs_curr.append(tr.obs[j])
                        obs_goal.append(tr.obs[goal_idx])
                        actions.append(tr.actions[j])

                        is_idle = tr.target_idx[j] == tr.curr_landmark_idx[j]  # stay around the same landmark
                        distance.append(self._calc_distance(j, goal_idx, is_idle))
                        train_distance.append(1)
                        locomotion_experience += 1

                    if self.visualize_trajectories[TmaxMode.LOCOMOTION] is None:
                        self.visualize_trajectories[TmaxMode.LOCOMOTION] = tr.obs[i:goal_idx + 1]

                    i = goal_idx

        for tr in trajectories[TmaxMode.EXPLORATION]:
            i = 0
            while i < len(tr) - 1:
                # goal from the "future"
                goal_idx = i + self.params.locomotion_max_trajectory
                goal_idx = min(goal_idx, len(tr) - 1)

                for j in range(i, goal_idx):
                    obs_curr.append(tr.obs[j])
                    obs_goal.append(tr.obs[goal_idx])
                    actions.append(tr.actions[j])
                    distance.append(self._calc_distance(j, goal_idx, is_idle=False))

                    train_distance.append(locomotion_experience <= 0)
                    exploration_experience += 1

                if self.visualize_trajectories[TmaxMode.EXPLORATION] is None:
                    self.visualize_trajectories[TmaxMode.EXPLORATION] = tr.obs[i:goal_idx + 1]

                i = goal_idx

            if exploration_experience >= locomotion_experience > 0:
                break

        if len(obs_curr) <= 0:
            # no new data
            return

        if len(self.obs_curr) <= 0:
            self.obs_curr = np.array(obs_curr)
            self.obs_goal = np.array(obs_goal)
            self.actions = np.array(actions, dtype=np.int32)
            self.distance = np.array(distance)
            self.train_distance = np.array(train_distance, dtype=np.int32)
        else:
            self.obs_curr = np.append(self.obs_curr, obs_curr, axis=0)
            self.obs_goal = np.append(self.obs_goal, obs_goal, axis=0)
            self.actions = np.append(self.actions, actions, axis=0)
            self.distance = np.append(self.distance, distance, axis=0)
            self.train_distance = np.append(self.train_distance, train_distance, axis=0)

        log.info('New experience: loco - %d, explore - %d', locomotion_experience, exploration_experience)

        self._discard_data()

        assert len(self.obs_curr) == len(self.obs_goal)
        assert len(self.obs_curr) == len(self.actions)
        assert len(self.obs_curr) == len(self.distance)
        assert len(self.obs_curr) == len(self.train_distance)

    def _discard_data(self):
        """Remove some data if the current buffer is too big."""
        target_size = self.params.locomotion_target_buffer_size
        if len(self.obs_curr) <= target_size:
            return

        self.shuffle_data()
        self.obs_curr = self.obs_curr[:target_size]
        self.obs_goal = self.obs_goal[:target_size]
        self.actions = self.actions[:target_size]
        self.distance = self.distance[:target_size]
        self.train_distance = self.train_distance[:target_size]

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
        self.distance = self.distance[chaos]
        self.train_distance = self.train_distance[chaos]
