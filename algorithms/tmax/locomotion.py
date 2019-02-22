import numpy as np
import tensorflow as tf

from algorithms.buffer import Buffer
from algorithms.encoders import make_encoder
from algorithms.env_wrappers import get_observation_space
from algorithms.tf_utils import placeholders_from_spaces, placeholder_from_space, dense
from algorithms.tmax.tmax_utils import TmaxMode
from algorithms.tmax.trajectory import Trajectory
from utils.distributions import CategoricalProbabilityDistribution
from utils.timing import Timing
from utils.utils import log, AttrDict


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
        self.params = params

        self.visualize_trajectories = {TmaxMode.EXPLORATION: None, TmaxMode.LOCOMOTION: None}

        # self.buffers = AttrDict({
        #     'exploration': Buffer(self.params.locomotion_target_buffer_size // 2 + 1),
        #     'locomotion': Buffer(self.params.locomotion_target_buffer_size // 2 + 1),
        #     'idle': Buffer(self.params.locomotion_target_buffer_size // 2 + 1),
        # })
        #
        # self.data = Buffer()

        self.obs_curr, self.obs_goal, self.actions = [], [], []

    @staticmethod
    def _calc_distance(obs_idx, goal_idx, is_idle):
        dist_frames = goal_idx - obs_idx
        near, far = 5, 10
        train = True
        if is_idle:
            is_far = 0
        else:
            if dist_frames <= near:
                is_far = 0
            elif dist_frames >= far:
                is_far = 1
            else:
                is_far = 0
                train = False

        return is_far, train

    @staticmethod
    def _get_mode(tr, idx):
        if tr.modes[idx] == TmaxMode.EXPLORATION:
            return 'exploration'
        elif tr.modes[idx] == TmaxMode.LOCOMOTION:
            is_idle = tr.target_idx[idx] == tr.curr_landmark_idx[idx]
            if is_idle:
                return 'idle'
            else:
                return 'locomotion'
        else:
            raise Exception(f'Unknown mode {tr.modes[idx]}')

    def _split_by_mode(self, episode_trajectories):
        trajectories = AttrDict({'exploration': [], 'locomotion': [], 'idle': []})
        for tr in episode_trajectories:
            if len(tr) <= 2:
                continue

            curr_tr = Trajectory(tr.env_idx)
            curr_tr.add_frame(tr, 0)
            curr_mode = self._get_mode(curr_tr, 0)

            for i in range(1, len(tr)):
                mode = self._get_mode(tr, i)
                if curr_mode != mode:
                    # start new trajectory because mode has changed
                    trajectories[curr_mode].append(curr_tr)
                    curr_mode = mode
                    curr_tr = Trajectory(tr.env_idx)

                curr_tr.add_frame(tr, i)

            if len(curr_tr) >= 1:
                trajectories[curr_mode].append(curr_tr)

        for mode, traj in trajectories.items():
            if len(traj) > 0:
                self.visualize_trajectories[mode] = traj[-1]

        return trajectories

    def _extract_exploration(self, trajectories):
        """Exploration trajectories between landmarks."""
        buffer = Buffer()

        def add_exploration(tr_, i_, goal_idx_):
            distance, train_distance = self._calc_distance(i_, goal_idx_, is_idle=False)

            buffer.add(
                obs_curr=tr_.obs[i_],
                obs_goal=tr_.obs[goal_idx_],
                action=tr_.actions[i_],
                distance=distance,

                # train distance using exploration experience only if we don't have enough locomotion data
                train_distance=(train_distance and len(self.buffers.locomotion) < 5000),
                train_actions=True,
            )

        for tr in trajectories:
            landmark_indices = []

            assert tr.is_landmark[0]
            for i in range(len(tr)):
                if tr.is_landmark[i]:
                    landmark_indices.append(i)

            for l_idx in range(1, len(landmark_indices)):
                l_prev = landmark_indices[l_idx - 1]
                l_next = landmark_indices[l_idx]

                assert l_next > l_prev

                if l_next - l_prev < 5:
                    continue  # trajectory is too short (probably noise)

                goal_idx = l_next
                i_start = max(l_prev, goal_idx - self.params.locomotion_max_trajectory)
                for i in range(i_start, goal_idx):
                    add_exploration(tr, i, goal_idx)

        if len(self.buffers.locomotion) < 5000:
            for tr in trajectories:
                i = 0
                while i < len(tr) - 1:
                    goal_idx = i + self.params.locomotion_max_trajectory
                    goal_idx = min(goal_idx, len(tr) - 1)
                    for j in range(i, goal_idx):
                        add_exploration(tr, j, goal_idx)

                    i = goal_idx

        return buffer

    def _extract_locomotion(self, trajectories, is_idle=False, max_trajectory=None):
        if max_trajectory is None:
            max_trajectory = self.params.locomotion_max_trajectory

        buffer = Buffer()

        for tr in trajectories:
            if len(tr) <= 2:
                continue

            target_changes = [0]
            for i in range(1, len(tr)):
                if tr.target_idx[i] != tr.target_idx[i - 1]:
                    target_changes.append(i)
            target_changes.append(len(tr))

            for targ_i in range(1, len(target_changes)):
                targ_prev = target_changes[targ_i - 1]
                targ_next = target_changes[targ_i]

                goal_idx = min(targ_next, len(tr) - 1)
                i_start = max(targ_prev, goal_idx - max_trajectory)
                for i in range(i_start, goal_idx):
                    distance, train_distance = self._calc_distance(i, goal_idx, is_idle=is_idle)

                    buffer.add(
                        obs_curr=tr.obs[i],
                        obs_goal=tr.obs[goal_idx],
                        action=tr.actions[i],
                        distance=distance,
                        train_distance=train_distance,
                        train_actions=not is_idle,
                    )

        return buffer

    def _extract_idle(self, trajectories):
        if len(self.buffers.locomotion) < 25000:
            return Buffer()
        return self._extract_locomotion(trajectories, is_idle=True, max_trajectory=10000)

    def _extract_trajectories(self, trajectories, mode):
        if mode == 'exploration':
            buffer = self._extract_exploration(trajectories)
        elif mode == 'locomotion':
            buffer = self._extract_locomotion(trajectories)
        elif mode == 'idle':
            buffer = self._extract_idle(trajectories)
        return buffer

    # TODO remove
    def extract_data_______________old(self, episode_trajectories):
        timing = Timing()
        with timing.timeit('split'):
            trajectories = self._split_by_mode(episode_trajectories)

        buffers = AttrDict()
        with timing.timeit('extract'):
            for mode in trajectories.keys():
                buffers[mode] = self._extract_trajectories(trajectories[mode], mode)

        for mode in self.visualize_trajectories.keys():
            self.visualize_trajectories[mode] = None

        with timing.timeit('append'):
            max_size = self.params.locomotion_target_buffer_size // 2
            for mode in self.buffers.keys():
                self.buffers[mode].add_buff(buffers[mode])
                self.buffers[mode].shuffle_data()
                self.buffers[mode].trim_at(max_size)

        with timing.timeit('finalize'):
            add_size = max_size
            min_size = min(len(self.buffers.exploration), len(self.buffers.locomotion), len(self.buffers.idle))
            if min_size > 5000:
                add_size = max(min_size, 100)

            sizes = {key: len(value) for key, value in self.buffers.items()}
            log.info('Add experience sizes: %r', sizes)

            self.data.clear()

            for mode, buffer in self.buffers.items():
                self.data.add_buff(buffer, add_size)

            self.data.shuffle_data()
            self.data.trim_at(self.params.locomotion_target_buffer_size)

        log.info('Timing %r', timing)

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

                locomotion_traj = all(mode == TmaxMode.LOCOMOTION for mode in trajectory.modes[l_prev:l_next])
                deliberate_actions = sum(trajectory.deliberate_action[l_prev:l_next]) == l_next - l_prev
                if not (locomotion_traj or deliberate_actions):
                    continue

                log.info('Locomotion trajectory!')

                assert l_next > l_prev
                traj_len = l_next - l_prev
                if traj_len > self.params.locomotion_max_trajectory:
                    # trajectory is too long and probably too noisy
                    log.info('Trajectory too long %d', traj_len)
                    continue
                else:
                    log.info('Trajectory okay %d', traj_len)

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
