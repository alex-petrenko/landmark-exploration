import copy
import math
import random
import time
from collections import deque
from functools import partial
from os.path import join

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from algorithms.agent import AgentLearner, TrainStatus
from algorithms.baselines.ppo.agent_ppo import PPOBuffer, AgentPPO
from algorithms.curiosity.ecr_map.ecr_map import ECRMapModule
from algorithms.distance.distance import DistanceBuffer
from algorithms.multi_env import MultiEnv
from algorithms.tmax.graph_encoders import make_graph_encoder
from algorithms.tmax.locomotion import LocomotionNetwork, LocomotionBuffer, LocomotionNetworkParams
from algorithms.tmax.navigator import Navigator
from algorithms.tmax.tmax_utils import TmaxMode, TmaxTrajectoryBuffer
from algorithms.topological_maps.map_builder import MapBuilder
from algorithms.topological_maps.topological_map import TopologicalMap, map_summaries
from algorithms.utils.algo_utils import EPS, num_env_steps, main_observation, goal_observation
from algorithms.utils.encoders import make_encoder, make_encoder_with_goal, get_enc_params
from algorithms.utils.env_wrappers import main_observation_space, is_goal_based_env
from algorithms.utils.models import make_model
from algorithms.utils.tf_utils import dense, count_total_parameters, placeholder_from_space, placeholders, \
    image_summaries_rgb, summary_avg_min_max, merge_summaries, tf_shape, placeholder
from utils.distributions import CategoricalProbabilityDistribution
from utils.envs.generate_env_map import generate_env_map
from utils.tensorboard import image_summary
from utils.timing import Timing
from utils.utils import log, AttrDict, numpy_all_the_way, model_dir, max_with_idx, ensure_dir_exists


class ActorCritic:
    def __init__(self, env, ph_observations, params, has_goal, name):
        with tf.variable_scope(name):
            obs_space = main_observation_space(env)

            self.ph_timer = tf.placeholder(tf.float32, shape=[None], name='ph_timer')
            timer = tf.expand_dims(self.ph_timer, axis=1)

            self.ph_observations = ph_observations
            self.ph_ground_truth_actions = placeholder_from_space(env.action_space)

            # placeholder for the goal observation (if available)
            self.ph_goal_obs = None
            self.has_goal = has_goal
            if self.has_goal:
                # goal obs has the same shape as main obs
                self.ph_goal_obs = placeholder_from_space(main_observation_space(env))

            make_encoder_func = make_encoder_with_goal if self.has_goal else make_encoder

            # placeholders for the topological map
            self.ph_neighbors, self.ph_num_neighbors = None, None
            if params.use_neighborhood_encoder:
                self.ph_num_neighbors = tf.placeholder(tf.int32, shape=[None])

            self.num_actions = env.action_space.n

            reg = None  # don't use L2 regularization

            # actor computation graph
            act_enc_params = get_enc_params(params, 'actor')
            act_encoder = tf.make_template(
                'act_obs_enc', make_encoder_func, create_scope_now_=True,
                obs_space=obs_space, regularizer=reg, enc_params=act_enc_params,
            )

            # use actor encoder as main observation encoder (including landmarks, etc.)
            if self.has_goal:
                act_goal_encoder = act_encoder(self.ph_observations, self.ph_goal_obs)
                act_encoded_obs = act_goal_encoder.encoded_input
                self.encode_single_obs = act_goal_encoder.encoder_obs.encoded_input
            else:
                act_encoded_obs = act_encoder(self.ph_observations).encoded_input
                self.encode_single_obs = act_encoded_obs

            self.encoded_obs_size = tf_shape(self.encode_single_obs)[-1]

            if params.use_neighborhood_encoder:
                self.ph_neighbors = placeholder([params.max_neighborhood_size, self.encoded_obs_size])
                act_neighborhood_encoder = make_graph_encoder(
                    self.ph_neighbors, self.ph_num_neighbors, params, 'act_graph_enc',
                )
                encoded_neighborhoods = act_neighborhood_encoder.encoded_neighborhoods
                act_obs_and_neighborhoods = tf.concat([act_encoded_obs, encoded_neighborhoods], axis=1)
            else:
                self.ph_neighbors = None
                act_obs_and_neighborhoods = act_encoded_obs

            if params.with_timer:
                act_obs_and_neighborhoods = tf.concat([act_obs_and_neighborhoods, timer], axis=1)
            actor_model = make_model(act_obs_and_neighborhoods, reg, params, 'act_mdl')

            actions_fc = dense(actor_model.latent, params.model_fc_size // 2, reg)
            self.action_logits = tf.contrib.layers.fully_connected(actions_fc, self.num_actions, activation_fn=None)
            self.best_action_deterministic = tf.argmax(self.action_logits, axis=1)
            self.actions_distribution = CategoricalProbabilityDistribution(self.action_logits)
            self.act = self.actions_distribution.sample()
            self.action_prob = self.actions_distribution.probability(self.act)

            # critic computation graph
            value_enc_params = get_enc_params(params, 'critic')
            value_encoder = tf.make_template(
                'val_obs_enc', make_encoder_func, create_scope_now_=True,
                obs_space=obs_space, regularizer=reg, enc_params=value_enc_params,
            )

            if self.has_goal:
                value_encoded_obs = value_encoder(self.ph_observations, self.ph_goal_obs).encoded_input
            else:
                value_encoded_obs = value_encoder(self.ph_observations).encoded_input

            if params.use_neighborhood_encoder:
                value_neighborhood_encoder = make_graph_encoder(
                    self.ph_neighbors, self.ph_num_neighbors, params, 'value_graph_enc',
                )
                encoded_neighborhoods = value_neighborhood_encoder.encoded_neighborhoods
                value_obs_and_neighborhoods = tf.concat([value_encoded_obs, encoded_neighborhoods], axis=1)
            else:
                value_obs_and_neighborhoods = value_encoded_obs

            if params.with_timer:
                value_obs_and_neighborhoods = tf.concat([value_obs_and_neighborhoods, timer], axis=1)
            value_model = make_model(value_obs_and_neighborhoods, reg, params, 'val_mdl')

            value_fc = dense(value_model.latent, params.model_fc_size // 2, reg)
            self.value = tf.squeeze(tf.contrib.layers.fully_connected(value_fc, 1, activation_fn=None), axis=[1])

            log.info('Total parameters so far: %d', count_total_parameters())

    def input_dict(self, observations, goals, neighbors_encoded, num_neighbors, timer):
        feed_dict = {self.ph_observations: observations, self.ph_timer: timer}
        if self.has_goal:
            feed_dict[self.ph_goal_obs] = goals
        if self.ph_neighbors is not None and self.ph_num_neighbors is not None:
            feed_dict[self.ph_neighbors] = neighbors_encoded
            feed_dict[self.ph_num_neighbors] = num_neighbors
        return feed_dict

    def invoke(self, session, observations, goals, neighbors_encoded, num_neighbors, timer, deterministic=False):
        ops = [
            self.best_action_deterministic if deterministic else self.act,
            self.action_prob,
            self.value,
        ]
        feed_dict = self.input_dict(observations, goals, neighbors_encoded, num_neighbors, timer)
        actions, action_prob, values = session.run(ops, feed_dict=feed_dict)
        return actions, action_prob, values

    def best_action(self, session, observations, goals, neighbors_encoded, num_neighbors, timer, deterministic=False):
        feed_dict = self.input_dict(observations, goals, neighbors_encoded, num_neighbors, timer)
        actions = session.run(self.best_action_deterministic if deterministic else self.act, feed_dict=feed_dict)
        return actions

    def encode_landmarks(self, session, landmarks):
        """This is mainly used to precalculate the landmark embeddings for graph encoder."""
        return session.run(self.encode_single_obs, feed_dict={self.ph_observations: landmarks})


class TmaxPPOBuffer(PPOBuffer):
    def __init__(self):
        super(TmaxPPOBuffer, self).__init__()
        self.neighbors, self.num_neighbors = None, None
        self.modes = None
        self.masks = None
        self.timer = None
        self.is_random = None

    def reset(self):
        super(TmaxPPOBuffer, self).reset()
        self.neighbors, self.num_neighbors = [], []
        self.modes = []
        self.masks = []
        self.timer = []
        self.is_random = []

    # noinspection PyMethodOverriding
    def add(
            self, obs, goals, actions, action_probs, rewards, dones,
            values, neighbors, num_neighbors, modes, masks, timer, is_random,
    ):
        """Append one-step data to the current batch of observations."""
        args = copy.copy(locals())
        super(TmaxPPOBuffer, self)._add_args(args)

    def split_by_mode(self):
        buffers = {}
        for mode in TmaxMode.all_modes():
            buffers[mode] = TmaxPPOBuffer()
            buffers[mode].reset()

        for i in range(len(self)):
            if self.is_random[i] or self.masks[i] == 0:
                continue

            mode = self.modes[i]
            for key, x in self.__dict__.items():
                if x is None or x.size == 0:
                    continue

                buffers[mode].__dict__[key].append(x[i])

        for mode in TmaxMode.all_modes():
            buffers[mode].to_numpy()

        return buffers


class TmaxManager:
    """
    This class takes care of topological memory and other aspects of policy execution and learning in the
    (potentially) multi-env setting.
    """

    def __init__(self, agent):
        self.initialized = False
        self._verbose = False

        self.agent = agent
        self.curiosity = agent.curiosity
        self.params = agent.params
        self.num_envs = self.params.num_envs

        self.navigator = Navigator(agent)

        # we need to potentially preserve a few most recent copies of the persistent map
        # because when we update the persistent map not all of the environments switch to it right away,
        # we might need to wait until the episode end in all of them
        self.dense_persistent_maps = deque([])
        self.sparse_persistent_maps = deque([])

        # references to current persistent maps associated with the env
        self.current_dense_maps = None
        self.current_sparse_maps = None

        self.dense_map_size_before_locomotion = 0
        self.sparse_map_size_before_locomotion = 0

        self.env_steps = 0
        self.episode_frames = [0] * self.num_envs
        self.end_episode = [self.params.exploration_budget * 1000] * self.num_envs

        # frame at which we switched to exploration mode
        self.exploration_started = [0] * self.num_envs

        # we collect some amount of random experience at the end of every exploration trajectory
        # to make the data for the distance network training more diverse
        self.random_mode = [False] * self.num_envs

        self.locomotion_targets = [None] * self.num_envs
        self.locomotion_final_targets = [None] * self.num_envs  # final target (e.g. goal observation)

        self.global_stage = TmaxMode.EXPLORATION
        self.last_stage_change = self.params.distance_bootstrap

        self.mode = [TmaxMode.EXPLORATION] * self.num_envs
        self.env_stage = [TmaxMode.EXPLORATION] * self.num_envs
        self.intrinsic_reward = [TmaxMode.EXPLORATION] * self.num_envs

        # if persistent map is provided, then we can skip the entire exploration stage
        self.stage_change_required = self.params.persistent_map_checkpoint is not None

        self.exploration_trajectories = deque([], maxlen=150)

        self.locomotion_success = deque([], maxlen=300)

    def initialize(self, obs, info, env_steps):
        if self.initialized:
            return

        self.env_steps = env_steps

        def empty_map():
            return TopologicalMap(obs[0], directed_graph=False, initial_info=info[0])

        self.dense_persistent_maps.append(empty_map())
        self.sparse_persistent_maps.append(empty_map())
        self._maybe_load_maps()

        self.current_dense_maps = []
        self.current_sparse_maps = []
        for i in range(self.num_envs):
            self.current_dense_maps.append(self.dense_persistent_maps[-1])
            self.current_sparse_maps.append(self.sparse_persistent_maps[-1])

        map_builder = MapBuilder(self.agent)
        map_builder.calc_distances_to_landmarks(self.sparse_persistent_maps[-1], self.dense_persistent_maps[-1])

        self.last_stage_change = max(self.last_stage_change, env_steps)

        for i in range(self.num_envs):
            self._new_episode(i)

        self.initialized = True
        return self.mode

    def _maybe_load_map(self, current_map, map_type):
        checkpoint_dir = model_dir(self.params.experiment_dir())
        map_dir = ensure_dir_exists(join(checkpoint_dir, map_type))
        current_map.maybe_load_checkpoint(map_dir)

    def _maybe_load_maps(self):
        self._maybe_load_map(self.dense_persistent_maps[-1], 'dense')
        self._maybe_load_map(self.sparse_persistent_maps[-1], 'sparse')

    def _save_map(self, current_map, map_type, is_sparse):
        checkpoint_dir = model_dir(self.params.experiment_dir())
        map_dir = ensure_dir_exists(join(checkpoint_dir, map_type))
        current_map.save_checkpoint(
            map_dir,
            map_img=self.agent.map_img, coord_limits=self.agent.coord_limits, verbose=True, is_sparse=is_sparse,
        )

    def save(self):
        if len(self.dense_persistent_maps) > 0:
            self._save_map(self.dense_persistent_maps[-1], 'dense', is_sparse=False)
        if len(self.sparse_persistent_maps) > 0:
            self._save_map(self.sparse_persistent_maps[-1], 'sparse', is_sparse=True)

    def _log_verbose(self, s, *args):
        if self._verbose:
            log.debug(s, *args)

    def is_episode_reset(self):
        reset = [False] * self.num_envs
        for env_i in range(self.num_envs):
            if self.end_episode[env_i] < 0:
                continue

            if self.episode_frames[env_i] >= self.end_episode[env_i]:
                reset[env_i] = True

        return reset

    def get_timer(self):
        timer = np.ones(self.num_envs, dtype=np.float32)
        for env_i in range(self.num_envs):
            exploration_mode = self.mode[env_i] == TmaxMode.EXPLORATION
            assert not exploration_mode or self.end_episode[env_i] >= self.params.exploration_budget

            if self.end_episode[env_i] < 0 or not exploration_mode:
                timer[env_i] = 1.0  # does not matter, because it's never used
            else:
                assert exploration_mode
                assert self.exploration_started[env_i] >= 0

                frames_so_far = self.episode_frames[env_i] - self.exploration_started[env_i]
                remaining_frames = self.params.exploration_budget - frames_so_far
                timer[env_i] = max(remaining_frames / self.params.exploration_budget, 0.0)
                assert -EPS < timer[env_i] < 1.0 + EPS

        return timer

    def save_trajectories(self, trajectories):
        """Save the last exploration trajectory for every env_idx to potentially later use as a demonstration."""
        for t in trajectories:
            is_all_exploration = all(stage == TmaxMode.EXPLORATION for stage in t.stage)
            if not is_all_exploration:
                continue

            # calculate total intrinsic reward over the trajectory
            total_intrinsic_reward = sum(r for r in t.intrinsic_reward)

            self.exploration_trajectories.append((total_intrinsic_reward, t))

            # if len(self.exploration_trajectories) < self.exploration_trajectories.maxlen:
            #     self.exploration_trajectories.append((total_intrinsic_reward, t))
            #     continue
            #
            # # if list is full then find trajectory with minimum intrinsic reward, remove from deque and append new
            # min_reward_idx = 0
            # for i, past_trajectory in enumerate(self.exploration_trajectories):
            #     reward, _ = past_trajectory
            #     if reward < self.exploration_trajectories[min_reward_idx][0]:
            #         min_reward_idx = i
            #
            # if total_intrinsic_reward > self.exploration_trajectories[min_reward_idx][0]:
            #     log.info('Found better exploration trajectory with reward %.3f', total_intrinsic_reward)
            #     del self.exploration_trajectories[min_reward_idx]
            #     self.exploration_trajectories.append((total_intrinsic_reward, t))

    def envs_with_locomotion_targets(self, env_indices):
        envs_with_goal, envs_without_goal = [], []
        for env_i in env_indices:
            locomotion_target_idx = self.locomotion_targets[env_i]
            if locomotion_target_idx is None:
                envs_without_goal.append(env_i)
            else:
                envs_with_goal.append(env_i)
        return envs_with_goal, envs_without_goal

    def get_locomotion_targets(self, env_indices):
        if len(env_indices) <= 0:
            return []

        targets = [None] * len(env_indices)

        for i, env_i in enumerate(env_indices):
            assert self.mode[env_i] == TmaxMode.LOCOMOTION
            locomotion_target_idx = self.locomotion_targets[env_i]
            if locomotion_target_idx is None:
                continue

            target_obs = self.current_dense_maps[env_i].get_observation(locomotion_target_idx)
            targets[i] = target_obs

        return targets

    def _get_locomotion_final_goal_locomotion_stage(self, env_i):
        """
        In locomotion stage we first go to randomly sampled location and then use random policy from there to
        collect trajectories for locomotion policy training.
        """
        m = self.current_dense_maps[env_i]
        nodes = list(m.graph.nodes)
        random_goal = random.choice(nodes)
        log.info('Locomotion final goal for locomotion is %d, env %d', random_goal, env_i)
        return random_goal

    def _get_locomotion_final_goal_exploration_stage(self, env_i):
        """Sample target according to UCB of value estimate."""
        curr_sparse_map = self.current_sparse_maps[env_i]

        # don't allow locomotion to most far away targets right away
        # this is to force exploration policy to find shorter routes to interesting locations
        total_frames = max(0, self.env_steps - self.params.distance_bootstrap)
        stage_idx = total_frames // (2 * self.params.stage_duration)
        max_distance = max(5, stage_idx * 1000)
        potential_targets = MapBuilder.sieve_landmarks_by_distance(curr_sparse_map, max_distance=max_distance)

        # calculate UCB of value estimate for all targets
        total_num_samples = 0
        for target in potential_targets:
            num_samples = curr_sparse_map.graph.nodes[target]['num_samples']
            total_num_samples += num_samples

        max_ucb = -math.inf
        max_ucb_target = -1
        for target in potential_targets:
            value = curr_sparse_map.graph.nodes[target]['value_estimate']
            num_samples = curr_sparse_map.graph.nodes[target]['num_samples']
            ucb_degree = 2.0  # exploration/exploitation tradeoff (TODO: move to params)
            ucb = value + ucb_degree * math.sqrt(math.log(total_num_samples) / num_samples)
            if ucb > max_ucb:
                max_ucb = ucb
                max_ucb_target = target

        # corresponding location in the dense map
        node_data = curr_sparse_map.graph.nodes[max_ucb_target]
        traj_idx = node_data.get('traj_idx', 0)
        frame_idx = node_data.get('frame_idx', 0)

        # log.debug('Location %d is max_ucb_target for exploration (t: %d, f: %d)', max_ucb_target, traj_idx, frame_idx)

        curr_dense_map = self.current_dense_maps[env_i]
        dense_map_landmark = curr_dense_map.frame_to_node_idx[traj_idx][frame_idx]

        # log.info('Sparse map node %d corresponds to dense map node %d', max_ucb_target, dense_map_landmark)

        locomotion_goal_idx = dense_map_landmark
        log.info(
            'Locomotion final goal for exploration is %d (%d) with value %.3f, samples %d and UCB %.3f',
            locomotion_goal_idx, max_ucb_target, node_data['value_estimate'], node_data['num_samples'], max_ucb,
        )
        node_data['num_samples'] += 1

        return locomotion_goal_idx

    def _get_locomotion_final_goal(self, env_i):
        if self.env_stage[env_i] == TmaxMode.LOCOMOTION:
            return self._get_locomotion_final_goal_locomotion_stage(env_i)
        else:
            return self._get_locomotion_final_goal_exploration_stage(env_i)

    @staticmethod
    def _pick_best_exploration_trajectory(agent, trajectories, curr_sparse_map):
        map_builder = MapBuilder(agent)

        max_dist_between_landmarks = 500

        trajectory_landmarks = [[] for _ in range(len(trajectories))]
        num_landmarks = [0] * len(trajectories)

        for i, t in enumerate(trajectories):
            log.debug('Processing trajectory %d with %d frames', i, len(t))

            if t.is_random[-1]:
                log.error('Last frame of trajectory %d is random!', i)
                log.error('%r', [r for r in t.is_random])
            assert not t.is_random[-1]

            m = copy.deepcopy(curr_sparse_map)
            m.new_episode()
            is_frame_a_landmark = map_builder.add_trajectory_to_sparse_map(m, t)
            landmark_frames = np.nonzero(is_frame_a_landmark)[0]

            trim_at = 0 if len(landmark_frames) <= 0 else landmark_frames[0]
            trim_at_landmark = 0
            for j in range(1, len(landmark_frames)):
                if landmark_frames[j] - landmark_frames[j - 1] < max_dist_between_landmarks:
                    trim_at = landmark_frames[j]
                    trim_at_landmark = j
                else:
                    break

            trajectories[i].trim_at(trim_at + 1)
            trajectory_landmarks[i] = landmark_frames[:trim_at_landmark + 1]
            num_landmarks[i] = len(trajectory_landmarks[i])

            log.debug(
                'Truncated trajectory %d to %d frames and %d landmarks',
                i, len(trajectories[i]), num_landmarks[i],
            )

        max_landmarks, best_trajectory_idx = max_with_idx(num_landmarks)
        log.debug(
            'Selected traj %d with %d landmarks and avg. distance of %.3f',
            best_trajectory_idx, max_landmarks, len(trajectories[best_trajectory_idx]) / max_landmarks + EPS,
        )

        return best_trajectory_idx, max_landmarks

    def _prepare_persistent_map_for_locomotion(self):
        """Pick an exploration trajectory and turn it into a dense persistent map."""
        log.warning('Prepare persistent map for locomotion!')

        if len(self.exploration_trajectories) <= 0:
            # we don't have any trajectories yet, need more exploration
            return False

        trajectory_rewards = [t[0] for t in self.exploration_trajectories]
        log.info('Best trajectories rewards: %r', trajectory_rewards)

        trajectories = [t[1] for t in self.exploration_trajectories]

        assert self.params.exploration_budget > self.params.random_frames_at_the_end

        # truncate trajectories
        for t_idx, t in enumerate(trajectories):
            first_exploration_frame = len(t)
            for i in range(len(t)):
                if t.mode[i] == TmaxMode.EXPLORATION:
                    first_exploration_frame = i
                    break

            t.trim_at(first_exploration_frame + self.params.max_exploration_trajectory)
            log.info('Trimmed trajectory %d at %d frames', t_idx, len(t))

        curr_sparse_map = copy.deepcopy(self.sparse_persistent_maps[-1])
        best_trajectory_idx, max_landmarks = self._pick_best_exploration_trajectory(
            self.agent, trajectories, curr_sparse_map,
        )

        if max_landmarks == 0:
            log.debug('Could not find any trajectory with nonzero novel landmarks')
            return False

        best_trajectory = trajectories[best_trajectory_idx]
        best_trajectory.save(self.params.experiment_dir())

        map_builder = MapBuilder(self.agent)
        best_trajectory = map_builder.sparsify_trajectory(best_trajectory)

        # reset exploration trajectories
        self.exploration_trajectories.clear()

        curr_dense_map = copy.deepcopy(self.dense_persistent_maps[-1])
        curr_sparse_map = copy.deepcopy(self.sparse_persistent_maps[-1])

        is_frame_a_landmark = map_builder.add_trajectory_to_sparse_map(curr_sparse_map, best_trajectory)
        landmark_frames = np.nonzero(is_frame_a_landmark)[0]
        log.debug('Added best trajectory to sparse map, landmark frames: %r', landmark_frames)

        self.sparse_persistent_maps.append(curr_sparse_map)
        self.sparse_map_size_before_locomotion = self.sparse_persistent_maps[-1].num_landmarks()

        new_dense_map = map_builder.add_trajectory_to_dense_map(curr_dense_map, best_trajectory)
        self.dense_persistent_maps.append(new_dense_map)
        self.dense_map_size_before_locomotion = self.dense_persistent_maps[-1].num_landmarks()

        map_builder.calc_distances_to_landmarks(curr_sparse_map, new_dense_map)
        map_builder.sieve_landmarks_by_distance(curr_sparse_map)  # for test

        # just in case
        new_dense_map.new_episode()
        curr_sparse_map.new_episode()

        log.info('Saving new persistent maps...')
        self.save()
        return True

    def _prepare_persistent_map_for_exploration(self):
        log.warning('Prepare persistent map for exploration!')
        new_dense_map = copy.deepcopy(self.dense_persistent_maps[-1])
        new_sparse_map = copy.deepcopy(self.sparse_persistent_maps[-1])

        # reset UCB statistics
        # for node in new_sparse_map.graph.nodes:
        #     new_sparse_map.graph.nodes[node]['num_samples'] = 1

        new_dense_map.new_episode()
        new_sparse_map.new_episode()

        self.dense_persistent_maps.append(new_dense_map)
        self.sparse_persistent_maps.append(new_sparse_map)

        log.debug('Prepared maps for exploration')

    def _update_value_estimates(self, m):
        if self.global_stage != TmaxMode.EXPLORATION:
            return

        landmark_observations = [m.get_observation(node) for node in m.graph.nodes]
        timer = [1.0 for _ in m.graph.nodes]
        _, _, values = self.agent.actor_critic.invoke(
            self.agent.session, landmark_observations, None, None, None, timer,  # does not work with goals!
        )

        assert len(values) == len(landmark_observations)
        for i, node in enumerate(m.graph.nodes):
            m.graph.nodes[node]['value_estimate'] = values[i]

    def _delete_old_maps(self, env_maps, maps):
        """Delete old persistent maps that aren't used anymore."""
        while len(maps) > 1:
            used_by_env = False
            for i in range(self.num_envs):
                used_by_env = used_by_env or env_maps[i] is maps[0]

            if not used_by_env:
                log.debug(
                    'Delete old persistent map with %d landmarks, it is not used anymore!',
                    maps[0].num_landmarks(),
                )
                maps.popleft()
            else:
                return

    def _new_episode(self, env_i):
        self.current_dense_maps[env_i] = self.dense_persistent_maps[-1]
        self.current_sparse_maps[env_i] = self.sparse_persistent_maps[-1]

        # Initialize curiosity episode map to be the current persistent map, this is to "push" the curious agent out
        # of the already explored region. Note - this only works with sparse ECR reward, otherwise the agent can get
        # stuck between two landmarks to maximize the immediate reward.
        if self.curiosity.episodic_maps is not None:
            self.curiosity.episodic_maps[env_i] = copy.deepcopy(self.sparse_persistent_maps[-1])
            self.curiosity.episodic_maps[env_i].new_episode()

        self.env_stage[env_i] = self.global_stage

        if env_i % 10 == 0:
            # we don't have to do it every time
            self._update_value_estimates(self.current_sparse_maps[env_i])

        self._delete_old_maps(self.current_dense_maps, self.dense_persistent_maps)
        self._delete_old_maps(self.current_sparse_maps, self.sparse_persistent_maps)

        self.episode_frames[env_i] = 0

        # this will be updated once locomotion goal is achieved (even if it's 0)
        self.end_episode[env_i] = self.params.exploration_budget * 1000
        self.exploration_started[env_i] = 0
        self.random_mode[env_i] = False

        self.locomotion_targets[env_i] = self.locomotion_final_targets[env_i] = None

        locomotion_goal = self._get_locomotion_final_goal(env_i)
        self.locomotion_final_targets[env_i] = locomotion_goal
        # noinspection PyTypeChecker
        self.locomotion_targets[env_i] = 0  # will be updated on the next frame

        self.mode[env_i] = TmaxMode.LOCOMOTION
        self.navigator.reset(env_i, self.current_dense_maps[env_i])

        assert self.locomotion_targets[env_i] is not None
        assert self.locomotion_final_targets[env_i] is not None

    def _update_stage(self, env_steps):
        if env_steps - self.last_stage_change > self.params.stage_duration or self.stage_change_required:
            if self.global_stage == TmaxMode.LOCOMOTION:
                self.global_stage = TmaxMode.EXPLORATION
                self._prepare_persistent_map_for_exploration()
                log.debug('Stage changed to Exploration')

                self.last_stage_change = env_steps
                if self.params.persistent_map_checkpoint is not None:
                    # we want to switch back to locomotion right away
                    self.stage_change_required = True
                else:
                    self.stage_change_required = False
            else:
                self.stage_change_required = False
                success = self._prepare_persistent_map_for_locomotion()
                if success:
                    self.global_stage = TmaxMode.LOCOMOTION
                    self.last_stage_change = env_steps
                    log.debug('Stage changed to Locomotion')
                else:
                    log.warning('Failed to switch stage to locomotion, environment not explored enough!')
                    # little hack to give us more time for exploration
                    self.last_stage_change += self.params.stage_duration // 5

        if self.stage_change_required:
            self._update_stage(env_steps)

    def _update_locomotion(self, next_obs):
        next_target, next_target_d = self.navigator.get_next_target(
            self.current_dense_maps, next_obs, self.locomotion_final_targets, self.episode_frames,
        )

        for env_i in range(self.num_envs):
            if self.mode[env_i] != TmaxMode.LOCOMOTION:
                continue
            if self.locomotion_final_targets[env_i] is None:
                continue

            self.locomotion_targets[env_i] = next_target[env_i]
            exploration_stage = self.env_stage[env_i] == TmaxMode.EXPLORATION

            end_locomotion, locomotion_success = False, False

            if next_target[env_i] is None:
                log.warning(
                    'Agent in env %d got lost in %d steps while trying to reach %d',
                    env_i, self.episode_frames[env_i], self.locomotion_final_targets[env_i],
                )
                end_locomotion = True

            since_last_made_progress = self.episode_frames[env_i] - self.navigator.last_made_progress[env_i]
            if since_last_made_progress > 100:
                log.warning(
                    'Agent in env %d did not make any progress in %d frames while trying to reach %d',
                    env_i, since_last_made_progress, self.locomotion_final_targets[env_i],
                )
                end_locomotion = True

            if self.episode_frames[env_i] > self.params.max_episode / 2:
                log.error(
                    'Takes too much time (%d) to get to %d',
                    self.episode_frames[env_i], self.locomotion_final_targets[env_i],
                )
                end_locomotion = True

            if self.navigator.current_landmarks[env_i] == self.locomotion_final_targets[env_i]:
                # reached the locomotion goal
                log.info(
                    'Env %d reached locomotion goal %d in %d steps',
                    env_i, self.locomotion_final_targets[env_i], self.episode_frames[env_i],
                )
                end_locomotion = True
                locomotion_success = True

            if end_locomotion:
                if exploration_stage:
                    self.mode[env_i] = TmaxMode.EXPLORATION
                    self.exploration_started[env_i] = self.episode_frames[env_i]
                else:
                    # after we reached locomotion goal we just collect random experience
                    self.random_mode[env_i] = True

                self.end_episode[env_i] = self.episode_frames[env_i] + self.params.exploration_budget
                if self.env_stage[env_i] == TmaxMode.EXPLORATION:
                    self.end_episode[env_i] += self.params.random_frames_at_the_end

                self.locomotion_targets[env_i] = self.locomotion_final_targets[env_i] = None
                self.locomotion_success.append(locomotion_success)
                log.info('End locomotion, frame %d, end %d', self.episode_frames[env_i], self.end_episode[env_i])

    def _update_curiosity(self, obs, next_obs, dones, infos):
        mask = []
        for env_i in range(self.num_envs):
            update_curiosity = self.env_stage[env_i] == TmaxMode.EXPLORATION
            if self.random_mode[env_i]:
                update_curiosity = False
            mask.append(update_curiosity)

        curiosity_bonus = self.curiosity.generate_bonus_rewards(
            self.agent.session, obs, next_obs, None, dones, infos, mask=mask,
        )
        return curiosity_bonus

    def update(self, obs, next_obs, rewards, dones, infos, env_steps, timing=None, verbose=False):
        self._verbose = verbose
        if timing is None:
            timing = Timing()

        assert len(obs) == len(self.current_dense_maps)

        self.env_steps = env_steps

        curiosity_bonus = np.zeros(self.num_envs)
        augmented_rewards = np.zeros(self.num_envs)
        done_flags = np.array(dones)

        if self.params.persistent_map_checkpoint is None:
            # run curiosity only if we need to discover the map, otherwise we don't need it (map is provided)
            with timing.add_time('curiosity'):
                curiosity_bonus = self._update_curiosity(obs, next_obs, dones, infos)

        with timing.add_time('update_locomotion'):
            self._update_locomotion(next_obs)

        with timing.add_time('new_episode'):
            for env_i in range(self.num_envs):
                if dones[env_i]:
                    self._new_episode(env_i)
                else:
                    self.episode_frames[env_i] += 1

            timer = self.get_timer()
            for env_i in range(self.num_envs):
                if timer[env_i] < EPS:
                    assert self.mode[env_i] == TmaxMode.EXPLORATION
                    if not self.random_mode[env_i]:
                        log.info('Environment %d exploration out of timer (%f)', env_i, timer[env_i])
                        self.random_mode[env_i] = True
                        done_flags[env_i] = True  # for RL purposes this is the end of the episode

        self._update_stage(env_steps)

        # combine final rewards and done flags
        for env_i in range(self.num_envs):
            if self.mode[env_i] == TmaxMode.EXPLORATION:
                augmented_rewards[env_i] = rewards[env_i] + curiosity_bonus[env_i]

            self.intrinsic_reward[env_i] = curiosity_bonus[env_i]

        return augmented_rewards, done_flags


class AgentTMAX(AgentLearner):
    """Agent based on PPO+TMAX algorithm."""

    class Params(
        AgentPPO.Params,
        ECRMapModule.Params,
        LocomotionNetworkParams,
    ):
        """Hyperparams for the algorithm and the training process."""

        def __init__(self, experiment_name):
            """Default parameter values set in ctor."""
            # calling all parent constructors
            AgentPPO.Params.__init__(self, experiment_name)
            ECRMapModule.Params.__init__(self)
            LocomotionNetworkParams.__init__(self)

            # TMAX-specific parameters
            self.use_neighborhood_encoder = False
            self.graph_enc_name = 'rnn'  # 'rnn', 'deepsets'
            self.max_neighborhood_size = 6  # max number of neighbors that can be fed into policy at every timestep
            self.graph_encoder_rnn_size = 128  # size of GRU layer in RNN neighborhood encoder
            self.with_timer = True

            self.rl_locomotion = False
            self.locomotion_dense_reward = True  # TODO remove?
            self.locomotion_reached_threshold = 0.075  # if distance is less than that, we reached the target
            self.reliable_path_probability = 0.4  # product of probs along the path for it to be considered reliable
            self.reliable_edge_probability = 0.1
            self.successful_traversal_frames = 50  # if we traverse an edge in less than that, we succeeded

            self.exploration_budget = 1000
            self.random_frames_at_the_end = 400
            self.max_exploration_trajectory = 900  # should be less than exploration budget
            self.max_episode = 20000

            self.locomotion_experience_replay = True

            self.stage_duration = 3000000

            self.distance_network_checkpoint = None
            self.persistent_map_checkpoint = None

            # summaries, etc.
            self.use_env_map = True

        @staticmethod
        def filename_prefix():
            return 'tmax_'

    def __init__(self, make_env_func, params):
        """Initialize PPO computation graph and some auxiliary tensors."""
        super(AgentTMAX, self).__init__(params)

        # separate global_steps
        self.actor_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='actor_step')
        self.critic_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='critic_step')

        if self.params.rl_locomotion:
            self.loco_actor_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='loco_actor_step')
            self.loco_critic_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='loco_critic_step')
            self.loco_her_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='loco_her_step')
        else:
            # TODO remove
            self.loco_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='loco_step')

        self.make_env_func = make_env_func
        env = make_env_func()  # we need the env to query observation shape, number of actions, etc.

        self.is_goal_env = is_goal_based_env(env)

        self.obs_shape = list(main_observation_space(env).shape)
        self.ph_observations = placeholder_from_space(main_observation_space(env))
        self.ph_actions = placeholder_from_space(env.action_space)  # actions sampled from the policy
        self.ph_advantages, self.ph_returns, self.ph_old_action_probs = placeholders(None, None, None)
        self.ph_masks = placeholder(None, tf.int32)  # to mask experience that does not come from RL policy

        self.actor_critic = ActorCritic(
            env, self.ph_observations, self.params, has_goal=is_goal_based_env(env), name='main',
        )

        if self.params.rl_locomotion:
            self.loco_actor_critic = ActorCritic(env, self.ph_observations, self.params, has_goal=True, name='loco')
        else:
            self.locomotion = LocomotionNetwork(env, params)

        self.encoded_landmark_size = self.actor_critic.encoded_obs_size
        if not self.params.use_neighborhood_encoder:
            self.encoded_landmark_size = 1

        self.curiosity = ECRMapModule(env, params)
        self.curiosity.distance_buffer = DistanceBuffer(params)

        # reuse distance network from the curiosity module
        self.distance = self.curiosity.distance

        env.close()

        self.objectives = self.add_ppo_objectives(
            self.actor_critic,
            self.ph_actions, self.ph_old_action_probs, self.ph_advantages, self.ph_returns, self.ph_masks,
            self.params, self.actor_step,
        )

        if self.params.rl_locomotion:
            self.loco_objectives = self.add_ppo_objectives(
                self.loco_actor_critic,
                self.ph_actions, self.ph_old_action_probs, self.ph_advantages, self.ph_returns, self.ph_masks,
                self.params, self.loco_actor_step,
            )

        # optimizers
        actor_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='actor_opt')
        self.train_actor = actor_opt.minimize(self.objectives.actor_loss, global_step=self.actor_step)

        critic_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='critic_opt')
        self.train_critic = critic_opt.minimize(self.objectives.critic_loss, global_step=self.critic_step)

        if self.params.rl_locomotion:
            loco_actor_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='loco_actor_opt')
            self.train_loco_actor = loco_actor_opt.minimize(
                self.loco_objectives.actor_loss, global_step=self.loco_actor_step,
            )

            loco_critic_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='loco_critic_opt')
            self.train_loco_critic = loco_critic_opt.minimize(
                self.loco_objectives.critic_loss, global_step=self.loco_critic_step,
            )

            loco_her_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='loco_her_opt')
            self.train_loco_her = loco_her_opt.minimize(
                self.loco_objectives.gt_actions_loss, global_step=self.loco_her_step,
            )
        else:
            # TODO remove
            loco_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='loco_opt')
            self.train_loco = loco_opt.minimize(
                self.locomotion.loss, global_step=self.loco_step,
            )

        # summaries
        self.add_summaries()

        self.actor_summaries = merge_summaries(collections=['actor'])
        self.critic_summaries = merge_summaries(collections=['critic'])

        if self.params.rl_locomotion:
            self.loco_actor_summaries = merge_summaries(collections=['loco_actor'])
            self.loco_critic_summaries = merge_summaries(collections=['loco_critic'])
            self.loco_her_summaries = merge_summaries(collections=['loco_her'])
        else:
            self.loco_summaries = merge_summaries(collections=['locomotion'])

        self.saver = tf.train.Saver(max_to_keep=3)

        all_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(all_vars, print_info=True)

        # auxiliary stuff not related to the computation graph
        self.tmax_mgr = TmaxManager(self)

        self._last_tmax_map_summary = 0  # timestamp of the latest tmax map summary

        if self.params.use_env_map:
            self.map_img, self.coord_limits = generate_env_map(make_env_func)

    @staticmethod
    def add_ppo_objectives(actor_critic, actions, old_action_probs, advantages, returns, masks, params, step):
        masks = tf.to_float(masks)
        num_rl_samples = tf.maximum(tf.reduce_sum(masks), EPS)  # to prevent division by 0

        action_probs = actor_critic.actions_distribution.probability(actions)
        prob_ratio = action_probs / old_action_probs  # pi / pi_old

        clip_ratio = params.ppo_clip_ratio
        clipped_advantages = tf.where(advantages > 0, advantages * clip_ratio, advantages / clip_ratio)

        clipped = tf.logical_or(prob_ratio > clip_ratio, prob_ratio < 1.0 / clip_ratio)
        clipped = tf.cast(clipped, tf.float32)

        # PPO policy gradient loss
        ppo_loss = -tf.minimum(prob_ratio * advantages, clipped_advantages)
        ppo_loss = ppo_loss * masks
        ppo_loss = tf.reduce_sum(ppo_loss) / num_rl_samples

        # penalize for inaccurate value estimation
        value_loss = tf.square(returns - actor_critic.value)
        value_loss = value_loss * masks
        value_loss = tf.reduce_sum(value_loss) / num_rl_samples

        # behavior cloning loss (when we have "ground truth" actions)
        gt_actions_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actor_critic.ph_ground_truth_actions, logits=actor_critic.action_logits,
        )
        gt_actions_loss = tf.reduce_mean(gt_actions_loss)

        # penalize the agent for being "too sure" about it's actions (to prevent converging to the suboptimal local
        # minimum too soon)
        entropy_losses = actor_critic.actions_distribution.entropy()

        # make sure entropy is maximized only for state-action pairs with non-clipped advantage
        entropy_losses = (1.0 - clipped) * entropy_losses * masks
        entropy_loss = -tf.reduce_mean(entropy_losses)
        entropy_loss_coeff = tf.train.exponential_decay(
            params.initial_entropy_loss_coeff, tf.cast(step, tf.float32), 10.0, 0.95, staircase=True,
        )
        entropy_loss_coeff = tf.maximum(entropy_loss_coeff, params.min_entropy_loss_coeff)
        entropy_loss = entropy_loss_coeff * entropy_loss

        # auxiliary quantities (for tensorboard, logging, early stopping)
        log_p_old = tf.log(old_action_probs + EPS)
        log_p = tf.log(action_probs + EPS)

        sample_kl = tf.reduce_sum((log_p_old - log_p) * masks) / num_rl_samples
        sample_entropy = tf.reduce_sum(-log_p * masks) / num_rl_samples
        clipped_fraction = tf.reduce_sum(clipped * masks) / num_rl_samples

        # only use entropy bonus if the policy is not close to max entropy
        max_entropy = actor_critic.actions_distribution.max_entropy()
        entropy_loss = tf.cond(sample_entropy > 0.8 * max_entropy, lambda: 0.0, lambda: entropy_loss)

        # final losses to optimize
        actor_loss = ppo_loss + entropy_loss
        critic_loss = value_loss
        gt_actions_loss = gt_actions_loss + entropy_loss

        return AttrDict(locals())

    def add_summaries(self):
        # summaries for the agent and the training process
        with tf.name_scope('obs_summaries'):
            image_summaries_rgb(self.ph_observations, collections=['actor'])
            if self.is_goal_env:
                image_summaries_rgb(self.actor_critic.ph_goal_obs, name='goal', collections=['actor'])

        self.add_ppo_summaries(self.actor_critic, self.objectives, self.actor_step, self.critic_step)

        if self.params.rl_locomotion:
            image_summaries_rgb(self.loco_actor_critic.ph_goal_obs, name='loco_goal', collections=['loco_actor'])

            self.add_ppo_summaries(
                self.loco_actor_critic, self.loco_objectives, self.loco_actor_step, self.loco_critic_step,
                'loco_actor', 'loco_critic',
            )
            with tf.name_scope('loco_her'):
                locomotion_scalar = partial(tf.summary.scalar, collections=['loco_her'])
                locomotion_scalar('actions_loss', self.loco_objectives.gt_actions_loss)
                locomotion_scalar('entropy', tf.reduce_mean(self.loco_actor_critic.actions_distribution.entropy()))
                locomotion_scalar('sample_kl', self.loco_objectives.sample_kl)
        else:
            image_summaries_rgb(self.locomotion.ph_obs_prev, name='loco_prev', collections=['locomotion'])
            image_summaries_rgb(self.locomotion.ph_obs_curr, name='loco_curr', collections=['locomotion'])
            image_summaries_rgb(self.locomotion.ph_obs_goal, name='loco_goal', collections=['locomotion'])

            with tf.name_scope('locomotion'):
                locomotion_scalar = partial(tf.summary.scalar, collections=['locomotion'])
                locomotion_scalar('loco_steps', self.locomotion.step)
                locomotion_scalar('loss', self.locomotion.loss)
                locomotion_scalar('entropy', tf.reduce_mean(self.locomotion.actions_distribution.entropy()))

    def add_ppo_summaries(self, actor_critic, obj, actor_step, critic_step, actor_scope='actor', critic_scope='critic'):
        with tf.name_scope(actor_scope):
            summary_avg_min_max('returns', self.ph_returns, collections=[actor_scope])
            summary_avg_min_max('adv', self.ph_advantages, collections=[actor_scope])

            actor_scalar = partial(tf.summary.scalar, collections=[actor_scope])
            actor_scalar('action_avg', tf.reduce_mean(tf.to_float(actor_critic.act)))
            actor_scalar('selected_action_avg', tf.reduce_mean(tf.to_float(self.ph_actions)))

            actor_scalar('entropy', tf.reduce_mean(actor_critic.actions_distribution.entropy()))
            actor_scalar('entropy_coeff', obj.entropy_loss_coeff)

            actor_scalar('actor_training_steps', actor_step)

            with tf.name_scope('ppo'):
                actor_scalar('sample_kl', obj.sample_kl)
                actor_scalar('sample_entropy', obj.sample_entropy)
                actor_scalar('clipped_fraction', obj.clipped_fraction)

            with tf.name_scope('losses'):
                actor_scalar('action_loss', obj.ppo_loss)
                actor_scalar('entropy_loss', obj.entropy_loss)
                actor_scalar('actor_loss', obj.actor_loss)

        with tf.name_scope(critic_scope):
            critic_scalar = partial(tf.summary.scalar, collections=[critic_scope])
            critic_scalar('value', tf.reduce_mean(actor_critic.value))
            critic_scalar('value_loss', obj.value_loss)
            critic_scalar('critic_training_steps', critic_step)

    def _maybe_print(self, step, env_step, avg_rewards, avg_length, fps, t):
        log.info('<====== Step %d, env step %.2fM ======>', step, env_step / 1e6)
        log.info('Avg FPS: %.1f', fps)
        log.info('Timing: %s', t)

        if math.isnan(avg_rewards) or math.isnan(avg_length):
            log.info('Need to gather more data to calculate avg. reward...')
            return

        log.info('Avg. %d episode length: %.3f', self.params.stats_episodes, avg_length)
        best_avg_reward = self.best_avg_reward.eval(session=self.session)
        log.info(
            'Avg. %d episode reward: %.3f (best: %.3f)',
            self.params.stats_episodes, avg_rewards, best_avg_reward,
        )

    def initialize_variables(self):
        checkpoint_dir = model_dir(self.params.experiment_dir())
        try:
            self.saver.restore(self.session, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir))
        except ValueError:
            log.info('Didn\'t find a valid restore point, start from scratch')
            self.session.run(tf.global_variables_initializer())

        # restore only distance network if we have checkpoint for it
        if self.params.distance_network_checkpoint is not None:
            log.debug('Restoring distance net variables from %s', self.params.distance_network_checkpoint)
            variables = slim.get_variables_to_restore()
            distance_net_variables = [v for v in variables if v.name.split('/')[0] == 'distance']
            distance_net_saver = tf.train.Saver(distance_net_variables)
            distance_net_saver.restore(
                self.session, tf.train.latest_checkpoint(self.params.distance_network_checkpoint),
            )
            self.curiosity.initialized = True
            log.debug('Done loading distance network from checkpoint!')

        log.debug('Computation graph is initialized!')

    def _save(self, step, env_steps):
        super()._save(step, env_steps)
        self.tmax_mgr.save()

    def _maybe_aux_summaries(self, env_steps, avg_reward, avg_length, fps):
        self._report_basic_summaries(fps, env_steps)

        if math.isnan(avg_reward) or math.isnan(avg_length):
            # not enough data to report yet
            return

        summary = tf.Summary()
        summary.value.add(tag='0_aux/avg_reward', simple_value=float(avg_reward))
        summary.value.add(tag='0_aux/avg_length', simple_value=float(avg_length))

        # if it's not "initialized" yet, just don't report anything to tensorboard
        initial_best, best_reward = self.session.run([self.initial_best_avg_reward, self.best_avg_reward])
        if best_reward != initial_best:
            summary.value.add(tag='0_aux/best_reward_ever', simple_value=float(best_reward))

        self.summary_writer.add_summary(summary, env_steps)
        self.summary_writer.flush()

    def _maybe_tmax_summaries(self, tmax_mgr, env_steps):
        time_since_last = time.time() - self._last_tmax_map_summary
        tmax_map_summary_rate_seconds = 180
        if time_since_last > tmax_map_summary_rate_seconds:
            dense_map_summary_start = time.time()
            map_summaries(
                [tmax_mgr.dense_persistent_maps[-1]],
                env_steps, self.summary_writer, 'tmax_dense_map', self.map_img, self.coord_limits, is_sparse=False,
            )
            dense_map_summary_took = time.time() - dense_map_summary_start
            sparse_map_summary_start = time.time()
            map_summaries(
                [tmax_mgr.sparse_persistent_maps[-1]],
                env_steps, self.summary_writer, 'tmax_sparse_map', self.map_img, self.coord_limits, is_sparse=True,
            )
            sparse_map_summary_took = time.time() - sparse_map_summary_start
            log.info('Tmax map summaries took %.3f dense %.3f sparse', dense_map_summary_took, sparse_map_summary_took)
            self._last_tmax_map_summary = time.time()

        summary_obj = tf.Summary()

        if len(tmax_mgr.locomotion_success) > 0:
            summary_obj.value.add(
                tag='locomotion/locomotion_success_rate', simple_value=np.mean(tmax_mgr.locomotion_success),
            )

        summary_obj.value.add(
            tag='tmax_maps/dense_map_size_before_locomotion',
            simple_value=np.mean(tmax_mgr.dense_map_size_before_locomotion),
        )
        summary_obj.value.add(
            tag='tmax_maps/sparse_map_size_before_locomotion',
            simple_value=np.mean(tmax_mgr.sparse_map_size_before_locomotion),
        )

        summary_obj.value.add(tag='tmax/global_stage', simple_value=tmax_mgr.global_stage)
        summary_obj.value.add(tag='tmax/avg_mode', simple_value=np.mean(tmax_mgr.mode))
        summary_obj.value.add(tag='tmax/avg_env_stage', simple_value=np.mean(tmax_mgr.env_stage))

        self._landmark_summaries(self.tmax_mgr.dense_persistent_maps[-1], env_steps)

        self.summary_writer.add_summary(summary_obj, env_steps)
        self.summary_writer.flush()

    def _landmark_summaries(self, m, env_steps):
        """Observation summaries for the current persistent map."""
        logged_landmarks = []
        summary_writer = self.summary_writer

        def landmark_summary(idx, tag):
            obs = m.get_observation(idx)
            obs_summary = image_summary(obs, f'tmax_landmarks/landmark_{tag}')
            summary_writer.add_summary(obs_summary, env_steps)
            logged_landmarks.append(idx)

        all_landmarks = list(m.graph.nodes)
        landmark_last = all_landmarks[-1]
        if landmark_last not in logged_landmarks:
            landmark_summary(landmark_last, 'last')

        random.shuffle(all_landmarks)
        for node in all_landmarks:
            if node not in logged_landmarks:
                landmark_summary(node, 'random')
                break

    def _maybe_trajectory_summaries(self, trajectory_buffer, env_steps):
        time_since_last = time.time() - self._last_trajectory_summary
        if time_since_last < self.params.gif_save_rate or not trajectory_buffer.complete_trajectories:
            return

        start_gif_summaries = time.time()

        self._last_trajectory_summary = time.time()
        num_envs = self.params.gif_summary_num_envs

        trajectories = []
        trajectories_locomotion = []
        sq_sz = 5  # size of square to indicate TmaxMode in gifs

        for trajectory in trajectory_buffer.complete_trajectories[:num_envs]:
            img_array = numpy_all_the_way(trajectory.obs)[:, :, :, -3:]
            for i in range(img_array.shape[0]):
                if trajectory.is_random[i]:
                    color = [0, 0, 255]  # blue for random actions
                else:
                    if trajectory.mode[i] == TmaxMode.LOCOMOTION:
                        color = [255, 0, 0]  # red for locomotion
                    elif trajectory.mode[i] == TmaxMode.EXPLORATION:
                        color = [0, 255, 0]  # green for exploration
                    else:
                        raise NotImplementedError('Unknown TMAX mode. Use EXPLORATION or LOCOMOTION')

                img_array[i, -sq_sz:, -sq_sz:] = color

            if all(mode == TmaxMode.LOCOMOTION for mode in trajectory.mode):
                trajectories_locomotion.append(img_array)
            else:
                trajectories.append(img_array)

        if len(trajectories) > 0:
            self._write_gif_summaries(tag='obs_trajectories', gif_images=trajectories, step=env_steps)

        if len(trajectories_locomotion) > 0:
            self._write_gif_summaries(tag='loco_trajectories', gif_images=trajectories_locomotion, step=env_steps)

        log.info('Took %.3f seconds to write gif summaries', time.time() - start_gif_summaries)

    def best_action(self, observation):
        raise NotImplementedError('Use best_action_tmax instead')

    def best_action_tmax(self, observations, goals, deterministic=False):
        actions = self.actor_critic.best_action(
            self.session, observations, goals, None, None, deterministic,
        )
        return actions[0]

    def _locomotion_policy_step(self, env_i, obs_prev, observations, goals, actions, masks, is_random, tmax_mgr):
        if len(env_i) <= 0:
            return

        envs_with_goal, envs_without_goal = tmax_mgr.envs_with_locomotion_targets(env_i)

        goal_obs = tmax_mgr.get_locomotion_targets(envs_with_goal)
        assert len(goal_obs) == len(envs_with_goal)

        masks[env_i] = 0
        deterministic = False if random.random() < 0.1 else True

        if len(envs_with_goal) > 0:
            is_random[envs_with_goal] = 0
            goals[envs_with_goal] = goal_obs  # replace goals with locomotion goals
            actions[envs_with_goal] = self.locomotion.navigate(
                self.session,
                obs_prev[envs_with_goal], observations[envs_with_goal], goals[envs_with_goal],
                deterministic=deterministic,
            )
            for env_index in envs_with_goal:
                if actions[env_index] == 0:
                    # discourage idle actions to avoid getting stuck
                    if random.random() < 0.03:
                        actions[env_index] = np.random.randint(0, self.actor_critic.num_actions)

        if len(envs_without_goal) > 0:
            # use random actions
            # TODO: better random trajectories? (repeat actions for more frames, etc.?)
            actions[envs_without_goal] = np.random.randint(0, self.actor_critic.num_actions, len(envs_without_goal))
            is_random[envs_without_goal] = 1

    def _exploration_policy_step(
            self, env_i, observations, goals, neighbors, num_neighbors,
            actions, action_probs, values, masks, timer, is_random, tmax_mgr,
    ):
        if len(env_i) <= 0:
            return

        env_random, env_non_random = [], []
        for env_index in env_i:
            random_action = False if self.curiosity.is_initialized() else True
            if tmax_mgr.random_mode[env_index]:
                random_action = True

            if random_action:
                env_random.append(env_index)
            else:
                env_non_random.append(env_index)

        assert len(env_random) + len(env_non_random) == len(env_i)

        if len(env_non_random) > 0:
            goals_policy = goals[env_non_random] if self.is_goal_env else None
            neighbors_policy = num_neighbors_policy = None
            if neighbors is not None:
                neighbors_policy = neighbors[env_non_random]
                num_neighbors_policy = num_neighbors[env_non_random]

            actions[env_non_random], action_probs[env_non_random], values[env_non_random] = self.actor_critic.invoke(
                self.session, observations[env_non_random],
                goals_policy, neighbors_policy, num_neighbors_policy, timer[env_non_random],
            )
            is_random[env_non_random] = 0
            masks[env_non_random] = 1

        if len(env_random) > 0:
            actions[env_random] = np.random.randint(0, self.actor_critic.num_actions, len(env_random))
            is_random[env_random] = 1
            masks[env_random] = 0

    def policy_step(self, obs_prev, observations, goals, neighbors, num_neighbors):
        """Run exploration or locomotion policy depending on the state of the particular environment."""
        tmax_mgr = self.tmax_mgr
        num_envs = len(observations)

        modes = np.zeros(num_envs, np.int32)
        env_indices = {TmaxMode.IDLE_EXPLORATION: [], TmaxMode.LOCOMOTION: [], TmaxMode.EXPLORATION: []}
        for env_i, mode in enumerate(tmax_mgr.mode):
            env_indices[mode].append(env_i)
            modes[env_i] = mode
        total_num_indices = sum(len(v) for v in env_indices.values())
        assert total_num_indices == num_envs

        obs_prev = np.asarray(obs_prev)
        observations = np.asarray(observations)
        goals = np.asarray(goals)
        if neighbors is not None:
            neighbors = np.asarray(neighbors)
            num_neighbors = np.asarray(num_neighbors)

        actions = np.empty(num_envs, np.int32)
        action_probs = np.ones(num_envs, np.float32)
        values = np.zeros(num_envs, np.float32)
        masks = np.zeros(num_envs, np.int32)
        is_random = np.zeros(num_envs, np.uint8)
        timer = tmax_mgr.get_timer()

        self._locomotion_policy_step(
            env_indices[TmaxMode.LOCOMOTION], obs_prev, observations, goals, actions, masks, is_random, tmax_mgr,
        )

        self._exploration_policy_step(
            env_indices[TmaxMode.EXPLORATION], observations, goals, neighbors, num_neighbors,
            actions, action_probs, values, masks, timer, is_random, tmax_mgr,
        )

        return actions, action_probs, values, masks, goals, modes, timer, is_random

    def _get_observations(self, env_obs):
        """
        Split the dictionary returned by the environment into main and goal observation.
        Return actual goal observation if we're in a goal-based environment, otherwise return an empty numpy array
        as goal (just to simplify and unify the rest of the code.
        """
        main_obs, goal_obs = main_observation(env_obs), goal_observation(env_obs)
        if goal_obs is None:
            if not hasattr(self, 'fake_goal'):
                self.fake_goal = [np.empty_like(main_obs[0])] * len(main_obs)
            goal_obs = self.fake_goal

        return main_obs, goal_obs

    def _train_actor(self, buffer, env_steps, objectives, actor_critic, train_actor, actor_step, actor_summaries):
        """Train actor for multiple epochs on all collected experience."""
        summary = None
        step = actor_step.eval(session=self.session)
        if len(buffer) <= 0:
            return step

        for epoch in range(self.params.ppo_epochs):
            buffer.shuffle()
            sample_kl = []  # sample kl divergences for all mini batches in buffer

            for i in range(0, len(buffer), self.params.batch_size):
                with_summaries = self._should_write_summaries(step) and summary is None
                summaries = [actor_summaries] if with_summaries else []

                start, end = i, i + self.params.batch_size

                policy_input = actor_critic.input_dict(
                    buffer.obs[start:end], buffer.goals[start:end],
                    buffer.neighbors[start:end], buffer.num_neighbors[start:end],
                    buffer.timer[start:end],
                )

                result = self.session.run(
                    [objectives.sample_kl, train_actor] + summaries,
                    feed_dict={
                        self.ph_actions: buffer.actions[start:end],
                        self.ph_old_action_probs: buffer.action_probs[start:end],
                        self.ph_advantages: buffer.advantages[start:end],
                        self.ph_returns: buffer.returns[start:end],
                        self.ph_masks: buffer.masks[start:end],
                        **policy_input,
                    }
                )

                sample_kl.append(result[0])

                step += 1
                self._maybe_save(step, env_steps)

                if with_summaries:
                    summary = result[-1]
                    self.summary_writer.add_summary(summary, global_step=env_steps)

            mean_sample_kl = np.mean(sample_kl)
            if mean_sample_kl > self.params.target_kl:
                log.info(
                    'Early stopping after %d/%d epochs because of high KL divergence %.4f > %.4f (%s)',
                    epoch + 1, self.params.ppo_epochs, mean_sample_kl, self.params.target_kl, actor_step.name,
                )
                break

        return step

    def _train_critic(self, buffer, env_steps, objectives, actor_critic, train_critic, critic_step, critic_summaries):
        summary = None
        step = critic_step.eval(session=self.session)

        if len(buffer) <= 0:
            return

        prev_loss = 1e10
        for epoch in range(self.params.ppo_epochs):
            losses = []
            buffer.shuffle()

            for i in range(0, len(buffer), self.params.batch_size):
                with_summaries = self._should_write_summaries(step) and summary is None
                summaries = [critic_summaries] if with_summaries else []

                start, end = i, i + self.params.batch_size

                policy_input = actor_critic.input_dict(
                    buffer.obs[start:end], buffer.goals[start:end],
                    buffer.neighbors[start:end], buffer.num_neighbors[start:end],
                    buffer.timer[start:end],
                )

                result = self.session.run(
                    [objectives.critic_loss, train_critic] + summaries,
                    feed_dict={
                        self.ph_returns: buffer.returns[start:end],
                        self.ph_masks: buffer.masks[start:end],
                        **policy_input
                    },
                )

                step += 1
                losses.append(result[0])

                if with_summaries:
                    summary = result[-1]
                    self.summary_writer.add_summary(summary, global_step=env_steps)

            # check loss improvement at the end of each epoch, early stop if necessary
            avg_loss = np.mean(losses)
            if avg_loss >= prev_loss:
                log.info(
                    'Stopping after %d epochs because critic did not improve (%s)', epoch + 1, critic_step.name,
                )
                log.info('Was %.4f now %.4f, ratio %.3f', prev_loss, avg_loss, avg_loss / prev_loss)
                break

            prev_loss = avg_loss

    def _maybe_train_locomotion(self, data, env_steps):
        """Train locomotion using hindsight experience replay."""
        num_epochs = self.params.locomotion_experience_replay_epochs

        summary = None
        prev_loss = 1e10
        batch_size = self.params.locomotion_experience_replay_batch
        locomotion = self.locomotion
        loco_step = locomotion.step.eval(session=self.session)

        log.info('Training loco_her %d pairs, batch %d, epochs %d', len(data.buffer), batch_size, num_epochs)

        for epoch in range(num_epochs):
            losses = []

            obs_prev, obs_curr, obs_goal = data.buffer.obs_prev, data.buffer.obs_curr, data.buffer.obs_goal
            actions = data.buffer.actions

            for i in range(0, len(obs_curr) - 1, batch_size):
                # noinspection PyProtectedMember
                with_summaries = self._should_write_summaries(loco_step) and summary is None
                summaries = [self.loco_summaries] if with_summaries else []

                start, end = i, i + batch_size

                objectives = [locomotion.loss, locomotion.train_loco]

                result = self.session.run(
                    objectives + summaries,
                    feed_dict={
                        locomotion.ph_obs_prev: obs_prev[start:end],
                        locomotion.ph_obs_curr: obs_curr[start:end],
                        locomotion.ph_obs_goal: obs_goal[start:end],
                        locomotion.ph_actions: actions[start:end],
                        locomotion.ph_is_training: True,
                    }
                )

                loco_step += 1
                # noinspection PyProtectedMember
                self._maybe_save(loco_step, env_steps)

                losses.append(result[0])

                if with_summaries:
                    summary = result[-1]
                    self.summary_writer.add_summary(summary, global_step=env_steps)

            # check loss improvement at the end of each epoch, early stop if necessary
            avg_loss = np.mean(losses)
            if avg_loss >= prev_loss:
                log.info('Stopping loco_her after %d epochs because locomotion did not improve', epoch)
                log.info('Was %.4f now %.4f, ratio %.3f', prev_loss, avg_loss, avg_loss / prev_loss)
                break

            prev_loss = avg_loss

    def _train_tmax(self, step, buffer, locomotion_buffer, env_steps, timing):
        if self.curiosity.is_initialized():
            with timing.timeit('split_buffers'):
                buffers = buffer.split_by_mode()
                buffer.reset()  # discard the original data (before mode split)

            if self.params.persistent_map_checkpoint is None:
                # persistent map is not provided - train exploration policy to discover it online
                with timing.timeit('train_policy'):
                    log.info('Exploration buffer size %d', len(buffers[TmaxMode.EXPLORATION]))
                    max_buffer_size = self.params.rollout * self.params.num_envs
                    if len(buffers[TmaxMode.EXPLORATION]) > 0.1 * max_buffer_size:
                        step = self._train_actor(
                            buffers[TmaxMode.EXPLORATION], env_steps,
                            self.objectives, self.actor_critic, self.train_actor, self.actor_step,
                            self.actor_summaries,
                        )
                        self._train_critic(
                            buffers[TmaxMode.EXPLORATION], env_steps,
                            self.objectives, self.actor_critic, self.train_critic, self.critic_step,
                            self.critic_summaries,
                        )

        # Locomotion with reinforcement learning - currently not supported
        # if self.params.rl_locomotion:
        #     with timing.timeit('loco_rl'):
        #         self._train_actor(
        #             buffers[TmaxMode.LOCOMOTION], env_steps,
        #             self.loco_objectives, self.loco_actor_critic, self.train_loco_actor,
        #             self.loco_actor_step, self.loco_actor_summaries,
        #         )
        #         self._train_critic(
        #             buffers[TmaxMode.LOCOMOTION], env_steps,
        #             self.loco_objectives, self.loco_actor_critic, self.train_loco_critic,
        #             self.loco_critic_step, self.loco_critic_summaries,
        #         )

        # train locomotion with self imitation from trajectories
        with timing.timeit('train_loco'):
            if len(locomotion_buffer.buffer) >= self.params.locomotion_experience_replay_buffer:
                self._maybe_train_locomotion(locomotion_buffer, env_steps)
                locomotion_buffer.reset()
            else:
                log.info('Locomotion buffer size: %d', len(locomotion_buffer.buffer))

        if self.params.distance_network_checkpoint is None:
            # distance net not provided - train distance metric online
            with timing.timeit('train_curiosity'):
                self.curiosity.train(buffer, env_steps, agent=self)

        return step

    def _learn_loop(self, multi_env):
        """Main training loop."""
        step, env_steps = self.session.run([self.actor_step, self.total_env_steps])

        observations, goals = self._get_observations(multi_env.reset())
        obs_prev = observations
        infos = multi_env.info()

        buffer = TmaxPPOBuffer()

        # separate buffer for complete episode trajectories
        trajectory_buffer = TmaxTrajectoryBuffer(multi_env.num_envs)
        self.curiosity.set_trajectory_buffer(trajectory_buffer)

        locomotion_buffer = LocomotionBuffer(self.params) if self.params.locomotion_experience_replay else None

        tmax_mgr = self.tmax_mgr
        tmax_mgr.initialize(observations, infos, env_steps)

        def end_of_training(s, es):
            return s >= self.params.train_for_steps or es > self.params.train_for_env_steps

        while not end_of_training(step, env_steps):
            # collecting experience
            timing = Timing()
            num_steps = 0
            batch_start = time.time()
            with timing.timeit('experience'):
                buffer.reset()
                for rollout_step in range(self.params.rollout):
                    with timing.add_time('policy'):
                        actions, action_probs, values, masks, policy_goals, modes, timer, is_random = self.policy_step(
                            obs_prev, observations, goals, None, None,
                        )

                    # wait for all the workers to complete an environment step
                    with timing.add_time('env_step'):
                        reset = tmax_mgr.is_episode_reset()
                        env_obs, rewards, dones, infos = multi_env.step(actions, reset)

                    trajectory_buffer.add(observations, actions, infos, dones, tmax_mgr=tmax_mgr, is_random=is_random)

                    new_obs, new_goals = self._get_observations(env_obs)

                    with timing.add_time('tmax'):
                        rewards, done_flags = tmax_mgr.update(
                            observations, new_obs, rewards, dones, infos, env_steps, timing,
                        )

                    # add experience from all environments to the current buffer(s)
                    buffer.add(
                        observations, policy_goals, actions, action_probs,
                        rewards, done_flags, values,
                        None, None, modes, masks, timer, is_random,
                    )

                    obs_prev = observations
                    observations, goals = new_obs, new_goals

                    self.process_infos(infos)
                    num_steps_delta = num_env_steps(infos)
                    num_steps += num_steps_delta
                    env_steps += num_steps_delta

                # last step values are required for TD-return calculation
                _, _, values, *_ = self.policy_step(obs_prev, observations, goals, None, None)
                buffer.values.append(values)

            # calculate discounted returns and GAE
            buffer.finalize_batch(self.params.gamma, self.params.gae_lambda)

            if locomotion_buffer is not None:
                locomotion_buffer.extract_data(trajectory_buffer.complete_trajectories)

            with timing.timeit('train'):
                step = self._train_tmax(step, buffer, locomotion_buffer, env_steps, timing)

            with timing.timeit('summaries'):
                with timing.timeit('tmax_summaries'):
                    self._maybe_tmax_summaries(tmax_mgr, env_steps)
                with timing.timeit('traj_summaries'):
                    self._maybe_trajectory_summaries(trajectory_buffer, env_steps)
                with timing.timeit('cov_summaries'):
                    self._maybe_coverage_summaries(env_steps)
                with timing.timeit('curiosity_summaries'):
                    self.curiosity.additional_summaries(
                        env_steps, self.summary_writer, self.params.stats_episodes,
                        map_img=self.map_img, coord_limits=self.coord_limits,
                    )

            avg_reward = multi_env.calc_avg_rewards(n=self.params.stats_episodes)
            avg_length = multi_env.calc_avg_episode_lengths(n=self.params.stats_episodes)
            fps = num_steps / (time.time() - batch_start)

            self._maybe_print(step, env_steps, avg_reward, avg_length, fps, timing)
            self._maybe_update_avg_reward(avg_reward, multi_env.stats_num_episodes())
            self._maybe_aux_summaries(env_steps, avg_reward, avg_length, fps)

            tmax_mgr.save_trajectories(trajectory_buffer.complete_trajectories)
            trajectory_buffer.reset_trajectories()

    def learn(self):
        status = TrainStatus.SUCCESS
        multi_env = None
        try:
            multi_env = MultiEnv(
                self.params.num_envs,
                self.params.num_workers,
                make_env_func=self.make_env_func,
                stats_episodes=self.params.stats_episodes,
            )

            self._learn_loop(multi_env)
        except (Exception, KeyboardInterrupt, SystemExit):
            log.exception('Interrupt...')
            status = TrainStatus.FAILURE
        finally:
            log.info('Closing env...')
            if multi_env is not None:
                multi_env.close()

        return status

# TODO: branch off from locomotion trajectory?
# TODO: separate coverage for different stages
