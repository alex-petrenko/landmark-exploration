import copy
import math
import random
import time
from collections import deque
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from algorithms.agent import AgentLearner, TrainStatus
from algorithms.algo_utils import EPS, num_env_steps, main_observation, goal_observation
from algorithms.baselines.ppo.agent_ppo import PPOBuffer
from algorithms.encoders import make_encoder, make_encoder_with_goal
from algorithms.env_wrappers import main_observation_space, is_goal_based_env
from algorithms.models import make_model
from algorithms.multi_env import MultiEnv
from algorithms.tf_utils import dense, count_total_parameters, placeholder_from_space, placeholders, \
    image_summaries_rgb, summary_avg_min_max, merge_summaries, tf_shape, placeholder
from algorithms.tmax.graph_encoders import make_graph_encoder
from algorithms.tmax.landmarks_encoder import LandmarksEncoder
from algorithms.tmax.locomotion import LocomotionNetwork, LocomotionBuffer
from algorithms.tmax.reachability import ReachabilityNetwork, ReachabilityBuffer
from algorithms.tmax.tmax_utils import TmaxMode
from algorithms.tmax.topological_map import TopologicalMap, get_position
from algorithms.tmax.trajectory import TrajectoryBuffer
from utils.distributions import CategoricalProbabilityDistribution
from utils.graph import visualize_graph_tensorboard
from utils.timing import Timing
from utils.utils import log, AttrDict, numpy_all_the_way, min_with_idx


class ActorCritic:
    def __init__(self, env, ph_observations, params, name):
        with tf.variable_scope(name):
            obs_space = env.observation_space

            self.ph_observations = ph_observations

            # placeholder for the goal observation (if available)
            self.ph_goal_obs = None
            self.is_goal_env = is_goal_based_env(env)
            if self.is_goal_env:
                # goal obs has the same shape as main obs
                self.ph_goal_obs = placeholder_from_space(main_observation_space(env))

            make_encoder_func = make_encoder_with_goal if self.is_goal_env else make_encoder

            # placeholders for the topological map
            self.ph_neighbors, self.ph_num_neighbors = None, None
            if params.use_neighborhood_encoder:
                self.ph_num_neighbors = tf.placeholder(tf.int32, shape=[None])

            self.num_actions = env.action_space.n

            reg = None  # don't use L2 regularization

            # actor computation graph
            act_encoder = tf.make_template(
                'act_obs_enc', make_encoder_func, create_scope_now_=True,
                obs_space=obs_space, regularizer=reg, params=params,
            )

            # use actor encoder as main observation encoder (including landmarks, etc.)
            if self.is_goal_env:
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

            actor_model = make_model(act_obs_and_neighborhoods, reg, params, 'act_mdl')

            actions_fc = dense(actor_model.latent, params.model_fc_size // 2, reg)
            action_logits = tf.contrib.layers.fully_connected(actions_fc, self.num_actions, activation_fn=None)
            self.best_action_deterministic = tf.argmax(action_logits, axis=1)
            self.actions_distribution = CategoricalProbabilityDistribution(action_logits)
            self.act = self.actions_distribution.sample()
            self.action_prob = self.actions_distribution.probability(self.act)

            # critic computation graph
            value_encoder = tf.make_template(
                'val_obs_enc', make_encoder_func, create_scope_now_=True,
                obs_space=obs_space, regularizer=reg, params=params,
            )
            if self.is_goal_env:
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

            value_model = make_model(value_obs_and_neighborhoods, reg, params, 'val_mdl')

            value_fc = dense(value_model.latent, params.model_fc_size // 2, reg)
            self.value = tf.squeeze(tf.contrib.layers.fully_connected(value_fc, 1, activation_fn=None), axis=[1])

            log.info('Total parameters so far: %d', count_total_parameters())

    def input_dict(self, observations, goals, neighbors_encoded, num_neighbors):
        feed_dict = {self.ph_observations: observations}
        if self.is_goal_env:
            feed_dict[self.ph_goal_obs] = goals
        if self.ph_neighbors is not None and self.ph_num_neighbors is not None:
            feed_dict[self.ph_neighbors] = neighbors_encoded
            feed_dict[self.ph_num_neighbors] = num_neighbors
        return feed_dict

    def invoke(self, session, observations, goals, neighbors_encoded, num_neighbors, deterministic=False):
        ops = [
            self.best_action_deterministic if deterministic else self.act,
            self.action_prob,
            self.value,
        ]
        feed_dict = self.input_dict(observations, goals, neighbors_encoded, num_neighbors)
        actions, action_prob, values = session.run(ops, feed_dict=feed_dict)
        return actions, action_prob, values

    def best_action(self, session, observations, goals, neighbors_encoded, num_neighbors, deterministic=False):
        feed_dict = self.input_dict(observations, goals, neighbors_encoded, num_neighbors)
        actions = session.run(self.best_action_deterministic if deterministic else self.act, feed_dict=feed_dict)
        return actions

    def encode_landmarks(self, session, landmarks):
        """This is mainly used to precalculate the landmark embeddings."""
        return session.run(self.encode_single_obs, feed_dict={self.ph_observations: landmarks})


class TmaxPPOBuffer(PPOBuffer):
    def __init__(self):
        super(TmaxPPOBuffer, self).__init__()
        self.neighbors, self.num_neighbors = None, None
        self.masks = None

    def reset(self):
        super(TmaxPPOBuffer, self).reset()
        self.neighbors, self.num_neighbors = [], []
        self.masks = []

    # noinspection PyMethodOverriding
    def add(self, obs, goals, actions, action_probs, rewards, dones, values, neighbors, num_neighbors, masks):
        """Append one-step data to the current batch of observations."""
        args = copy.copy(locals())
        super(TmaxPPOBuffer, self)._add_args(args)

    def split_by_mask(self):
        buffers = [TmaxPPOBuffer(), TmaxPPOBuffer()]
        for buffer in buffers:
            buffer.reset()

            for key in buffer.__dict__.keys():
                x = self.__dict__[key]
                if x is None or x.size == 0:
                    continue

                buffer.__dict__[key] = []

        for i in range(len(self)):
            for key, x in self.__dict__.items():
                if x is None or x.size == 0:
                    continue

                mask = self.masks[i]
                buffers[mask].__dict__[key].append(x[i])

        for buffer in buffers:
            for item, x in buffer.__dict__.items():
                if x is None:
                    continue
                buffer.__dict__[item] = np.asarray(x)

            for i in range(len(buffer)):
                buffer.masks[i] = 1

        return buffers


class TmaxManager:
    """
    This class takes care of topological memory and other aspects of policy execution and learning in the
    (potentially) multi-env setting.
    """

    def __init__(self, agent):
        self.initialized = False
        self._verbose = False
        self._is_bootstrap = False

        self.agent = agent
        self.params = agent.params
        self.num_envs = self.params.num_envs
        self.neighbors_buffer = np.zeros([
            self.params.num_envs, self.params.max_neighborhood_size, agent.encoded_landmark_size,
        ])
        self.maps = None
        self.episodic_maps = None

        self.landmarks_encoder = LandmarksEncoder(agent.actor_critic.encode_landmarks)

        self.episode_frames = [0] * self.num_envs
        self.locomotion_prev = [None] * self.num_envs  # starting landmark for locomotion policy
        self.locomotion_targets = [None] * self.num_envs  # immediate goal for locomotion policy
        self.locomotion_final_targets = [None] * self.num_envs  # final target (e.g. goal observation)
        self.last_locomotion_success = [0] * self.num_envs
        self.locomotion_achieved_goal = deque([])

        self.mode = [TmaxMode.EXPLORATION] * self.num_envs

        self.need_reset = [True] * self.num_envs

        self.new_landmark_threshold = self.params.new_landmark_threshold
        self.loop_closure_threshold = self.params.loop_closure_threshold

        self.idle_frames = [0] * self.num_envs
        self.action_frames = [np.random.randint(1, 3) for _ in range(self.num_envs)]

        self.deliberate_action = [True] * self.num_envs

        self.samples_per_mode = {
            TmaxMode.EXPLORATION: 0, TmaxMode.LOCOMOTION: 0, TmaxMode.IDLE_EXPLORATION: 0,
        }

        self.total_episode_bonus = np.zeros(self.num_envs)

    def initialize(self, obs, info):
        if self.initialized:
            return

        self.maps, self.episodic_maps = [], []
        for i in range(self.num_envs):
            self.maps.append(
                TopologicalMap(obs[i], self.params.directed_edges, info[i]))
            self.episodic_maps.append(
                TopologicalMap(obs[i], self.params.directed_edges, info[i]))

        self.initialized = True

    def _log_verbose(self, s, *args):
        if self._verbose:
            log.debug(s, *args)

    def _log_distances(self, env_i, neighbor_indices, distance):
        """Optional diagnostic logging."""
        log_reachability = True
        if self._verbose and log_reachability:
            neighbor_distance = {}
            for i, neighbor_idx in enumerate(neighbor_indices):
                neighbor_distance[neighbor_idx] = '{:.3f}'.format(distance[i])
            self._log_verbose('Env %d distance: %r', env_i, neighbor_distance)

    def _select_locomotion_target(self, env_i, verbose=False):
        if self.mode[env_i] != TmaxMode.LOCOMOTION:
            self.locomotion_prev[env_i] = None
            self.locomotion_targets[env_i] = None
            self.locomotion_final_targets[env_i] = None
            return

        final_target = self.locomotion_final_targets[env_i]
        assert final_target is not None

        m = self.maps[env_i]
        path = m.get_path(m.curr_landmark_idx, final_target)

        if verbose:
            log.info('Shortest path from %d to %d is %r', m.curr_landmark_idx, final_target, path)

        if m.curr_landmark_idx == final_target or path is None or len(path) <= 1:
            # reached the target, switch to exploration policy
            self.mode[env_i] = TmaxMode.EXPLORATION
            self.locomotion_prev[env_i] = None
            self.locomotion_targets[env_i] = None
            self.locomotion_final_targets[env_i] = None
        else:
            assert len(path) >= 2
            self.locomotion_prev[env_i] = m.curr_landmark_idx
            self.locomotion_targets[env_i] = path[1]  # next vertex on the path

    def get_locomotion_targets(self, env_indices):
        assert len(env_indices) > 0
        targets = []

        for env_i in env_indices:
            assert self.mode[env_i] == TmaxMode.LOCOMOTION
            locomotion_target_idx = self.locomotion_targets[env_i]
            assert locomotion_target_idx is not None
            target_obs = self.maps[env_i].landmarks[locomotion_target_idx]
            targets.append(target_obs)

        assert len(targets) == len(env_indices)
        return targets

    def _new_episode(self, env_i, obs, info, goal):
        self.total_episode_bonus[env_i] = 0

        self.episodic_maps[env_i].reset(obs, info)
        self.episodic_maps[env_i].new_episode()

        m = self.maps[env_i]
        if self.need_reset[env_i]:
            m.reset(obs, info)
            self.need_reset[env_i] = False
        m.new_episode()

        self.mode[env_i] = None

        self.locomotion_prev[env_i] = None
        self.locomotion_targets[env_i] = self.locomotion_final_targets[env_i] = None
        self.last_locomotion_success[env_i] = 0

        self.idle_frames[env_i] = 0
        self.action_frames[env_i] = np.random.randint(1, 3)

        self.episode_frames[env_i] = 0

        m.curr_landmark_idx = 0  # TODO! proper localization? we're assuming start at 0th landmark

        locomotion_goal_idx = m.curr_landmark_idx

        if goal is not None and self.params.use_locomotion:
            # find reachable landmark that is closest to goal image according to distance metric
            reachable_indices = m.reachable_indices(m.curr_landmark_idx)
            reachable_landmarks, goal_obs = [], []
            for reachable_idx in reachable_indices:
                reachable_landmarks.append(m.landmarks[reachable_idx])
                goal_obs.append(goal)

            distances = self.agent.reachability.distances(self.agent.session, reachable_landmarks, goal_obs)
            min_d, min_d_idx = min_with_idx(distances)

            if min_d > self.params.new_landmark_threshold:
                # landmark closest to the goal is still too far from the goal, in this case let's pick locomotion goal
                # randomly
                locomotion_goal_idx = random.choice(reachable_indices)
                log.info('Random locomotion goal is %d (min_d is %.3f)', locomotion_goal_idx, min_d)
            else:
                # landmark closest to the goal is quite close to the goal, let's go there
                locomotion_goal_idx = reachable_indices[min_d_idx]
                log.info('Final locomotion goal is %d (min_d is %.3f)', locomotion_goal_idx, min_d)

        if locomotion_goal_idx == m.curr_landmark_idx or random.random() < 0.1:
            self.mode[env_i] = TmaxMode.EXPLORATION
            self.locomotion_final_targets[env_i] = None
        else:
            self.mode[env_i] = TmaxMode.LOCOMOTION
            self.locomotion_final_targets[env_i] = locomotion_goal_idx
            self._select_locomotion_target(env_i, verbose=True)

        assert self.mode[env_i] is not None
        if self.mode[env_i] == TmaxMode.LOCOMOTION:
            assert not self.params.use_locomotion
            assert self.locomotion_prev[env_i] is not None
            assert self.locomotion_targets[env_i] is not None
            assert self.locomotion_final_targets[env_i] is not None

    def _distance_to_locomotion_targets(self, obs):
        loco_env_indices = []
        loco_curr_obs = []
        for env_i in range(self.num_envs):
            if self.mode[env_i] == TmaxMode.LOCOMOTION:
                loco_env_indices.append(env_i)
                loco_curr_obs.append(obs[env_i])

        if len(loco_env_indices) == 0:
            return []

        locomotion_targets = self.get_locomotion_targets(loco_env_indices)
        distances = self.agent.reachability.distances(self.agent.session, loco_curr_obs, locomotion_targets)

        all_distances = [None] * self.num_envs
        for i, env_i in enumerate(loco_env_indices):
            all_distances[env_i] = distances[i]
        return all_distances

    def _localize(self, obs, info, maps, persistent_map):
        bonuses = np.zeros([self.num_envs])
        locomotion_reward = np.zeros([self.num_envs])
        locomotion_done = np.zeros([self.num_envs], dtype=bool)

        # noise-filtering parameter, how many frames we need to wait before we change localization
        localize_frames = 4

        # create a batch of all neighborhood observations from all envs for fast processing on GPU
        neighborhood_obs, current_obs = [], []
        for env_i, m in enumerate(maps):
            neighbor_indices = m.neighbor_indices()
            neighborhood_obs.extend([m.landmarks[i] for i in neighbor_indices])
            current_obs.extend([obs[env_i]] * len(neighbor_indices))

        assert len(neighborhood_obs) == len(current_obs)

        # calculate reachability for all neighborhoods in all envs
        distances = self.agent.reachability.distances(self.agent.session, neighborhood_obs, current_obs)

        dist_to_target = self._distance_to_locomotion_targets(obs)

        new_landmark_candidates = []
        closest_landmark_idx = [-1] * self.num_envs

        j = 0
        for env_i, m in enumerate(maps):
            neighbor_indices = m.neighbor_indices()
            j_next = j + len(neighbor_indices)
            distance = distances[j:j_next]

            if persistent_map:
                self._log_distances(env_i, neighbor_indices, distance)

            # check if we're far enough from all landmarks in the neighborhood
            min_d, min_d_idx = min_with_idx(distance)
            closest_landmark_idx[env_i] = neighbor_indices[min_d_idx]

            if self.locomotion_targets[env_i] is not None and self.mode[env_i] == TmaxMode.LOCOMOTION:
                assert dist_to_target[env_i] is not None
                locomotion_reward[env_i] -= dist_to_target[env_i] * 0.1  # scaling factor

            if min_d >= self.new_landmark_threshold:
                # we're far enough from all obs in the neighborhood, might have found something new!
                new_landmark_candidates.append(env_i)
            else:
                # we're still sufficiently close to our neighborhood, but maybe "current landmark" has changed
                m.new_landmark_candidate_frames = 0

                # crude localization
                if all(lm == closest_landmark_idx[env_i] for lm in m.closest_landmarks[-localize_frames:]):
                    if closest_landmark_idx[env_i] != m.curr_landmark_idx:
                        m.set_curr_landmark(closest_landmark_idx[env_i])

                if persistent_map and self.mode[env_i] == TmaxMode.LOCOMOTION:
                    if m.curr_landmark_idx == self.locomotion_targets[env_i] and dist_to_target[env_i] < 0.3:
                        locomotion_reward[env_i] += 1.0
                        log.info(
                            'Locomotion net gets reward +1! (%.3f) dist %.3f, prev %d, goal %d, frame %d, took %d',
                            locomotion_reward[env_i], dist_to_target[env_i], self.locomotion_prev[env_i],
                            m.curr_landmark_idx, self.episode_frames[env_i],
                            self.episode_frames[env_i] - self.last_locomotion_success[env_i],
                        )
                        self.last_locomotion_success[env_i] = self.episode_frames[env_i]

                        locomotion_done[env_i] = True
                        self.locomotion_achieved_goal.append(1)
                        m.update_edge_traversal(self.locomotion_prev[env_i], m.curr_landmark_idx, 1)
                        self._select_locomotion_target(env_i)

            j = j_next

        del neighborhood_obs
        del current_obs

        # Agents in some environments discovered landmarks that are far away from all landmarks in the immediate
        # vicinity. There are two possibilities:
        # 1) A new landmark should be created and added to the graph
        # 2) We're close to some other vertex in the graph - we've found a "loop closure", a new edge in a graph

        non_neighborhood_obs = []
        non_neighborhoods = {}
        current_obs = []
        for env_i in new_landmark_candidates:
            m = maps[env_i]
            non_neighbor_indices = m.non_neighbor_indices()
            non_neighborhoods[env_i] = non_neighbor_indices
            non_neighborhood_obs.extend([m.landmarks[i] for i in non_neighbor_indices])
            current_obs.extend([obs[env_i]] * len(non_neighbor_indices))

        assert len(non_neighborhood_obs) == len(current_obs)

        # calculate reachability for all non-neighbors
        distances = []
        batch_size = 1024
        for i in range(0, len(non_neighborhood_obs), batch_size):
            start, end = i, i + batch_size
            distances_batch = self.agent.reachability.distances(
                self.agent.session, non_neighborhood_obs[start:end], current_obs[start:end],
            )
            distances.extend(distances_batch)

        j = 0
        for env_i in new_landmark_candidates:
            m = maps[env_i]
            non_neighbor_indices = non_neighborhoods[env_i]
            j_next = j + len(non_neighbor_indices)
            distance = distances[j:j_next]

            min_d, min_d_idx = math.inf, math.inf
            if len(distance) > 0:
                min_d, min_d_idx = min_with_idx(distance)

            if min_d < self.loop_closure_threshold:
                # current observation is close to some other landmark, "close the loop" by creating a new edge
                m.new_landmark_candidate_frames = 0

                closest_landmark_idx[env_i] = non_neighbor_indices[min_d_idx]

                # crude localization
                if all(lm == closest_landmark_idx[env_i] for lm in m.closest_landmarks[-localize_frames:]):
                    m.set_curr_landmark(closest_landmark_idx[env_i])

                    bonuses[env_i] += self.params.map_expansion_reward  # we found a new edge! Cool!

                    if persistent_map and self.mode[env_i] == TmaxMode.LOCOMOTION:
                        self._log_verbose('Change current landmark to %d (loop closure)', m.curr_landmark_idx)
                        self._select_locomotion_target(env_i)
                        locomotion_done[env_i] = True

            else:
                # vertex is relatively far away from all vertices in the graph, we've found a new landmark!
                if m.new_landmark_candidate_frames >= localize_frames:
                    new_landmark_idx = m.add_landmark(obs[env_i], info[env_i])
                    m.set_curr_landmark(new_landmark_idx)
                    bonuses[env_i] += self.params.map_expansion_reward  # we found a new vertex! Cool!

                    closest_landmark_idx[env_i] = new_landmark_idx
                    m.new_landmark_candidate_frames = 0

                    if persistent_map and self.mode[env_i] == TmaxMode.LOCOMOTION:
                        self._select_locomotion_target(env_i)
                        locomotion_done[env_i] = True
                else:
                    m.new_landmark_candidate_frames += 1

            j = j_next

        # update localization info
        for env_i in range(self.num_envs):
            assert closest_landmark_idx[env_i] >= 0
            maps[env_i].closest_landmarks.append(closest_landmark_idx[env_i])

        return bonuses, locomotion_reward, locomotion_done

    def update(self, obs, goals, dones, infos, is_bootstrap=False, verbose=False):
        """Omnipotent function for the management of topological maps and policy modes, as well as localization."""
        self._verbose = verbose
        self._is_bootstrap = is_bootstrap

        done_flags = np.zeros([self.num_envs], dtype=bool)

        assert len(obs) == len(self.maps)
        num_envs = len(self.maps)

        for env_i, m in enumerate(self.maps):
            self.samples_per_mode[self.mode[env_i]] += 1

            if is_bootstrap:
                self.need_reset[env_i] = True

            if self.mode[env_i] == TmaxMode.LOCOMOTION:
                if self.episode_frames[env_i] - self.last_locomotion_success[env_i] > 300 or dones[env_i]:
                    # we're not making progress towards the final goal, let's switch to exploration
                    self.mode[env_i] = TmaxMode.EXPLORATION
                    self.locomotion_achieved_goal.append(0)
                    m.update_edge_traversal(self.locomotion_prev[env_i], self.locomotion_targets[env_i], 0)
                    self.locomotion_targets[env_i] = self.locomotion_final_targets[env_i] = None
                    done_flags[env_i] = True
                    log.info('Locomotion unsuccessful, switch to exploration')

            if dones[env_i]:
                env_goal = None if goals is None else goals[env_i]
                self._new_episode(env_i, obs[env_i], infos[env_i], env_goal)
            else:
                self.episode_frames[env_i] += 1

        if is_bootstrap:
            # don't bother updating the graph when the reachability net isn't trained yet
            return np.zeros([num_envs]), np.zeros([num_envs]), done_flags

        _, locomotion_rewards, locomotion_done = self._localize(obs, infos, self.maps, persistent_map=True)
        bonuses, _, _ = self._localize(obs, infos, self.episodic_maps, persistent_map=False)
        self.total_episode_bonus += bonuses

        done_flags = np.logical_or(done_flags, locomotion_done)
        return bonuses, locomotion_rewards, done_flags

    def combine_rewards_and_dones(self, rewards, dones, bonuses, loco_rewards, done_flags):
        final_rewards = [0] * self.num_envs
        final_dones = [False] * self.num_envs

        for env_i in range(self.num_envs):
            if self.mode[env_i] == TmaxMode.LOCOMOTION:
                final_rewards[env_i] = loco_rewards[env_i]
                final_dones[env_i] = dones[env_i] or done_flags[env_i]
            elif self.mode[env_i] == TmaxMode.EXPLORATION:
                final_rewards[env_i] = rewards[env_i] + bonuses[env_i]
                final_dones[env_i] = dones[env_i] or done_flags[env_i]
            else:
                raise Exception(f'Mode {self.mode[env_i]} not supported')

        return final_rewards, final_dones

    def get_neighbors(self):
        if not self.params.use_neighborhood_encoder:
            return None, None

        neighbors_buffer = self.neighbors_buffer
        maps = self.maps

        neighbors_buffer.fill(0)
        num_neighbors = [0] * len(maps)
        landmark_env_idx = []

        neighbor_landmarks, neighbor_hashes = [], []

        for env_idx, m in enumerate(maps):
            n_indices = m.neighbor_indices()
            current_landmark_idx = n_indices[0]  # always keep the "current landmark"
            n_indices = n_indices[1:]

            # order of neighbors does not matter
            random.shuffle(n_indices)
            n_indices = [current_landmark_idx] + n_indices

            for i, n_idx in enumerate(n_indices):
                if i >= self.params.max_neighborhood_size:
                    log.warning(
                        'Too many neighbors %d, max encoded is %d. Had to ignore some neighbors.',
                        len(n_indices), self.params.max_neighborhood_size,
                    )
                    break

                neighbor_landmarks.append(m.landmarks[n_idx])
                neighbor_hashes.append(m.hashes[n_idx])
                landmark_env_idx.append((env_idx, i))
            num_neighbors[env_idx] = min(len(n_indices), self.params.max_neighborhood_size)

        # calculate embeddings in one big batch
        self.landmarks_encoder.encode(self.agent.session, neighbor_landmarks, neighbor_hashes)

        # populate the buffer using cached embeddings
        for i, neighbor_hash in enumerate(neighbor_hashes):
            env_idx, neighbor_idx = landmark_env_idx[i]
            neighbors_buffer[env_idx, neighbor_idx] = self.landmarks_encoder.encoded_landmarks[neighbor_hash]

        return neighbors_buffer, num_neighbors


class AgentTMAX(AgentLearner):
    """Agent based on PPO+TMAX algorithm."""

    class Params(AgentLearner.AgentParams):
        """Hyperparams for the algorithm and the training process."""

        def __init__(self, experiment_name):
            """Default parameter values set in ctor."""
            super(AgentTMAX.Params, self).__init__(experiment_name)

            self.gamma = 0.99  # future reward discount
            self.gae_lambda = 0.8
            self.rollout = 64
            self.num_envs = 192  # number of environments to collect the experience from
            self.num_workers = 16  # number of workers used to run the environments

            # actor-critic (encoders and models)
            self.image_enc_name = 'convnet_84px'
            self.model_fc_layers = 1
            self.model_fc_size = 256
            self.model_recurrent = False

            # ppo-specific
            self.ppo_clip_ratio = 1.1  # we use clip(x, e, 1/e) instead of clip(x, 1+e, 1-e) in the paper
            self.target_kl = 0.03
            self.batch_size = 512
            self.ppo_epochs = 10

            # components of the loss function
            self.initial_entropy_loss_coeff = 0.1
            self.min_entropy_loss_coeff = 0.002

            # TMAX-specific parameters
            self.use_neighborhood_encoder = False
            self.graph_enc_name = 'rnn'  # 'rnn', 'deepsets'
            self.max_neighborhood_size = 6  # max number of neighbors that can be fed into policy at every timestep
            self.graph_encoder_rnn_size = 128  # size of GRU layer in RNN neighborhood encoder

            self.reachable_threshold = 6  # num. of frames between obs, such that one is reachable from the other
            self.unreachable_threshold = 20  # num. of frames between obs, such that one is unreachable from the other
            self.reachability_target_buffer_size = 100000  # target number of training examples to store
            self.reachability_train_epochs = 1
            self.reachability_batch_size = 256

            self.directed_edges = False  # whether to add directed on undirected edges on landmark discovery
            self.new_landmark_threshold = 0.9  # condition for considering current observation a "new landmark"
            self.loop_closure_threshold = 0.7  # condition for graph loop closure (finding new edge)
            self.map_expansion_reward = 1.0  # reward for finding new vertex or new edge in the topological map

            self.locomotion_max_trajectory = 35  # max trajectory length to be utilized for locomotion training
            self.locomotion_target_buffer_size = 100000  # target number of (obs, goal, action) tuples to store
            self.locomotion_train_epochs = 1
            self.locomotion_batch_size = 256
            self.rl_locomotion = True
            self.use_locomotion = False

            self.bootstrap_env_steps = 1000 * 1000

            self.gif_save_rate = 100  # number of seconds to wait before saving another gif to tensorboard
            self.gif_summary_num_envs = 2

            self.show_automap = False

            # training process
            self.learning_rate = 1e-4
            self.train_for_steps = self.train_for_env_steps = 10 * 1000 * 1000 * 1000
            self.use_gpu = True
            self.initial_save_rate = 1000

        @staticmethod
        def filename_prefix():
            return 'tmax_'

    def __init__(self, make_env_func, params):
        """Initialize PPO computation graph and some auxiliary tensors."""
        super(AgentTMAX, self).__init__(params)

        # separate global_steps
        self.actor_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='actor_step')
        self.critic_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='critic_step')
        self.reach_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='reach_step')

        if self.params.rl_locomotion:
            self.loco_actor_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='loco_actor_step')
            self.loco_critic_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='loco_critic_step')
        else:
            self.locomotion_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='locomotion_step')

        self.make_env_func = make_env_func
        env = make_env_func()  # we need the env to query observation shape, number of actions, etc.

        self.is_goal_env = is_goal_based_env(env)

        self.obs_shape = list(main_observation_space(env).shape)
        self.ph_observations = placeholder_from_space(main_observation_space(env))
        self.ph_actions = placeholder_from_space(env.action_space)  # actions sampled from the policy
        self.ph_advantages, self.ph_returns, self.ph_old_action_probs = placeholders(None, None, None)
        self.ph_masks = placeholder(None, tf.int32)  # to mask experience that does not come from RL policy

        self.actor_critic = ActorCritic(env, self.ph_observations, self.params, 'main')

        if self.params.rl_locomotion:
            self.loco_actor_critic = ActorCritic(env, self.ph_observations, self.params, 'loco')
        else:
            self.locomotion = LocomotionNetwork(env, params)

        self.reachability = ReachabilityNetwork(env, params)

        if self.params.use_neighborhood_encoder is None:
            self.encoded_landmark_size = 1
        else:
            self.encoded_landmark_size = self.actor_critic.encoded_obs_size

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

        reach_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='reach_opt')
        self.train_reachability = reach_opt.minimize(self.reachability.loss, global_step=self.reach_step)

        if self.params.rl_locomotion:
            loco_actor_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='loco_actor_opt')
            self.train_loco_actor = loco_actor_opt.minimize(
                self.loco_objectives.actor_loss, global_step=self.loco_actor_step,
            )

            loco_critic_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='loco_critic_opt')
            self.train_loco_critic = loco_critic_opt.minimize(
                self.loco_objectives.critic_loss, global_step=self.loco_critic_step,
            )
        else:
            locomotion_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='locomotion_opt')
            self.train_locomotion = locomotion_opt.minimize(self.locomotion.loss, global_step=self.locomotion_step)

        # summaries
        self.add_summaries()

        self.actor_summaries = merge_summaries(collections=['actor'])
        self.critic_summaries = merge_summaries(collections=['critic'])
        self.reach_summaries = merge_summaries(collections=['reachability'])

        if self.params.rl_locomotion:
            self.loco_actor_summaries = merge_summaries(collections=['loco_actor'])
            self.loco_critic_summaries = merge_summaries(collections=['loco_critic'])
        else:
            self.locomotion_summaries = merge_summaries(collections=['locomotion'])

        self.saver = tf.train.Saver(max_to_keep=3)

        all_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(all_vars, print_info=True)

        # auxiliary stuff not related to the computation graph
        self.tmax_mgr = TmaxManager(self)
        self._last_trajectory_summary = 0  # timestamp of the latest trajectory summary written

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

        return AttrDict(locals())

    def add_summaries(self):
        # summaries for the agent and the training process
        with tf.name_scope('obs_summaries'):
            image_summaries_rgb(self.ph_observations, collections=['actor'])
            if self.is_goal_env:
                image_summaries_rgb(self.actor_critic.ph_goal_obs, collections=['actor'])

        self.add_ppo_summaries(self.actor_critic, self.objectives, self.actor_step, self.critic_step)

        with tf.name_scope('reachability'):
            reachability_scalar = partial(tf.summary.scalar, collections=['reachability'])
            reachability_scalar('reach_loss', self.reachability.reach_loss)
            reachability_scalar('reach_correct', self.reachability.correct)
            reachability_scalar('reg_loss', self.reachability.reg_loss)

        if self.params.rl_locomotion:
            self.add_ppo_summaries(
                self.loco_actor_critic, self.loco_objectives, self.loco_actor_step, self.loco_critic_step,
                'loco_actor', 'loco_critic',
            )
        else:
            with tf.name_scope('locomotion'):
                locomotion_scalar = partial(tf.summary.scalar, collections=['locomotion'])
                locomotion_scalar('actions_loss', self.locomotion.actions_loss)
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

        log.info('Avg. %d episode lenght: %.3f', self.params.stats_episodes, avg_length)
        best_avg_reward = self.best_avg_reward.eval(session=self.session)
        log.info(
            'Avg. %d episode reward: %.3f (best: %.3f)',
            self.params.stats_episodes, avg_rewards, best_avg_reward,
        )

    def _maybe_aux_summaries(self, tmax_mgr, env_steps, avg_reward, avg_length, fps, bonuses):
        self._report_basic_summaries(fps, env_steps)

        if math.isnan(avg_reward) or math.isnan(avg_length):
            # not enough data to report yet
            return

        summary = tf.Summary()
        summary.value.add(tag='0_aux/avg_reward', simple_value=float(avg_reward))
        summary.value.add(tag='0_aux/avg_length', simple_value=float(avg_length))
        summary.value.add(tag='0_aux/avg_bonus_per_frame', simple_value=float(np.mean(bonuses)))
        summary.value.add(tag='0_aux/avg_bonus_per_episode', simple_value=float(np.mean(tmax_mgr.total_episode_bonus)))

        # if it's not "initialized" yet, just don't report anything to tensorboard
        initial_best, best_reward = self.session.run([self.initial_best_avg_reward, self.best_avg_reward])
        if best_reward != initial_best:
            summary.value.add(tag='0_aux/best_reward_ever', simple_value=float(best_reward))

        self.summary_writer.add_summary(summary, env_steps)
        self.summary_writer.flush()

    def _maybe_tmax_summaries(self, tmax_mgr, env_steps):
        maps = tmax_mgr.maps

        num_landmarks = [len(m.landmarks) for m in maps]
        num_neighbors = [len(m.neighbor_indices()) for m in maps]
        num_edges = [m.num_edges() for m in maps]

        avg_num_landmarks = sum(num_landmarks) / len(num_landmarks)
        avg_num_neighbors = sum(num_neighbors) / len(num_neighbors)
        avg_num_edges = sum(num_edges) / len(num_edges)

        summary_obj = tf.Summary()

        def map_summary(tag, value):
            summary_obj.value.add(tag=f'map/{tag}', simple_value=float(value))

        map_summary('avg_landmarks', avg_num_landmarks)
        map_summary('max_landmarks', max(num_landmarks))
        map_summary('avg_neighbors', avg_num_neighbors)
        map_summary('max_neighbors', max(num_neighbors))
        map_summary('avg_edges', avg_num_edges)
        map_summary('max_edges', max(num_edges))

        # locomotion summaries
        achieved_goal = tmax_mgr.locomotion_achieved_goal
        if len(achieved_goal) >= 300:
            while len(achieved_goal) > 300:
                achieved_goal.popleft()
            locomotion_avg_success = sum(achieved_goal) / len(achieved_goal)
            summary_obj.value.add(tag='locomotion/avg_success', simple_value=float(locomotion_avg_success))

        self.summary_writer.add_summary(summary_obj, env_steps)

        map_for_summary = random.choice(maps)
        random_graph_summary = visualize_graph_tensorboard(map_for_summary.to_nx_graph(), tag='map/random_graph')
        self.summary_writer.add_summary(random_graph_summary, env_steps)

        max_graph_idx = 0
        for i, m in enumerate(maps):
            if len(m.landmarks) > len(maps[max_graph_idx].landmarks):
                max_graph_idx = i

        max_graph_summary = visualize_graph_tensorboard(maps[max_graph_idx].to_nx_graph(), tag='map/max_graph')
        self.summary_writer.add_summary(max_graph_summary, env_steps)

        self.summary_writer.flush()

    def _maybe_trajectory_summaries(self, trajectory_buffer, env_steps):
        time_since_last = time.time() - self._last_trajectory_summary
        if time_since_last < self.params.gif_save_rate or not trajectory_buffer.complete_trajectories:
            return

        start_gif_summaries = time.time()

        self._last_trajectory_summary = time.time()
        num_envs = self.params.gif_summary_num_envs

        trajectories = [
            numpy_all_the_way(t.obs)[:, :, :, -3:] for t in trajectory_buffer.complete_trajectories[:num_envs]
        ]
        self._write_gif_summaries(tag='obs_trajectories', gif_images=trajectories, step=env_steps)
        log.info('Took %.3f seconds to write gif summaries', time.time() - start_gif_summaries)

    def best_action(self, observation):
        raise NotImplementedError('Use best_action_tmax instead')

    def best_action_tmax(self, observations, goals, deterministic=False):
        neighbors, num_neighbors = self.tmax_mgr.get_neighbors()
        actions = self.actor_critic.best_action(
            self.session, observations, goals, neighbors, num_neighbors, deterministic,
        )
        return actions[0]

    def policy_step(self, observations, goals, neighbors, num_neighbors, is_bootstrap):
        """Run exploration or locomotion policy depending on the state of the particular environment."""
        tmax_mgr = self.tmax_mgr
        num_envs = len(observations)

        env_indices = {TmaxMode.IDLE_EXPLORATION: [], TmaxMode.LOCOMOTION: [], TmaxMode.EXPLORATION: []}
        for env_i, mode in enumerate(tmax_mgr.mode):
            env_indices[mode].append(env_i)
        total_num_indices = sum(len(v) for v in env_indices.values())
        assert total_num_indices == num_envs

        observations = np.asarray(observations)
        if goals is not None:
            goals = np.asarray(goals)
        if neighbors is not None:
            neighbors = np.asarray(neighbors)
            num_neighbors = np.asarray(num_neighbors)

        actions = np.empty(num_envs, np.int32)
        action_probs = np.ones(num_envs, np.float32)
        values = np.zeros(num_envs, np.float32)
        masks = np.zeros(num_envs, np.int32)

        # IDLE_EXPLORATION policy (explore + idle) is used only for Montezuma
        env_i = env_indices[TmaxMode.IDLE_EXPLORATION]
        if len(env_i) > 0:
            masks[env_i] = 0  # don't train with ppo

            non_idle_i = []
            for env_index in env_i:
                # idle-random policy
                assert tmax_mgr.action_frames[env_index] > 0 or tmax_mgr.idle_frames[env_index] > 0

                if tmax_mgr.idle_frames[env_index] > 0:
                    # idle action
                    tmax_mgr.deliberate_action[env_index] = False
                    actions[env_index] = 0  # NOOP
                    tmax_mgr.idle_frames[env_index] -= 1
                    if tmax_mgr.idle_frames[env_index] <= 0:
                        tmax_mgr.action_frames[env_index] = np.random.randint(1, self.params.unreachable_threshold)
                else:
                    tmax_mgr.deliberate_action[env_index] = True
                    non_idle_i.append(env_index)

                    tmax_mgr.action_frames[env_index] -= 1
                    if tmax_mgr.action_frames[env_index] <= 0:
                        if random.random() < 0.5:
                            tmax_mgr.idle_frames[env_index] = np.random.randint(1, self.params.unreachable_threshold)
                        else:
                            tmax_mgr.idle_frames[env_index] = np.random.randint(1, 500)

            non_idle_i = np.asarray(non_idle_i)
            if is_bootstrap and len(non_idle_i) > 0:
                # just sample random action to save time
                num_actions = self.actor_critic.num_actions
                actions[non_idle_i] = np.random.randint(0, num_actions, len(non_idle_i))
            elif len(non_idle_i) > 0:
                neighbors_policy = num_neighbors_policy = None
                if neighbors is not None:
                    neighbors_policy = neighbors[non_idle_i]
                    num_neighbors_policy = num_neighbors[non_idle_i]
                actions[non_idle_i], action_probs[non_idle_i], values[non_idle_i] = self.actor_critic.invoke(
                    self.session, observations[non_idle_i], goals[non_idle_i],
                    neighbors_policy, num_neighbors_policy, deterministic=False,
                )

        env_i = env_indices[TmaxMode.LOCOMOTION]
        if len(env_i) > 0:
            masks[env_i] = 0
            goal_obs = tmax_mgr.get_locomotion_targets(env_i)
            assert len(goal_obs) == len(env_i)
            for env_index in env_i:
                tmax_mgr.deliberate_action[env_index] = True

            if self.params.rl_locomotion:
                actions[env_i], action_probs[env_i], values[env_i] = self.loco_actor_critic.invoke(
                    self.session, observations[env_i], goal_obs, None, None,
                )
            else:
                actions[env_i] = self.locomotion.navigate(
                    self.session, observations[env_i], goal_obs, deterministic=False,
                )

        env_i = env_indices[TmaxMode.EXPLORATION]
        if len(env_i) > 0:
            masks[env_i] = 1
            goals_policy = goals[env_i] if self.is_goal_env else None
            neighbors_policy = num_neighbors_policy = None
            if neighbors is not None:
                neighbors_policy = neighbors[env_i]
                num_neighbors_policy = num_neighbors[env_i]
            actions[env_i], action_probs[env_i], values[env_i] = self.actor_critic.invoke(
                self.session, observations[env_i], goals_policy, neighbors_policy, num_neighbors_policy,
            )
            for env_index in env_i:
                tmax_mgr.deliberate_action[env_index] = True

        return actions, action_probs, values, masks

    def _train_actor(self, buffer, env_steps, objectives, actor_critic, train_actor, actor_step, actor_summaries):
        # train actor for multiple epochs on all collected experience
        summary = None
        actor_step = actor_step.eval(session=self.session)
        if len(buffer) <= 0:
            return actor_step

        kl_running_avg = 0.0
        early_stop = False

        for epoch in range(self.params.ppo_epochs):
            buffer.shuffle()

            for i in range(0, len(buffer), self.params.batch_size):
                with_summaries = self._should_write_summaries(actor_step) and summary is None
                summaries = [actor_summaries] if with_summaries else []

                start, end = i, i + self.params.batch_size

                policy_input = actor_critic.input_dict(
                    buffer.obs[start:end], buffer.goals[start:end],
                    buffer.neighbors[start:end], buffer.num_neighbors[start:end],
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

                actor_step += 1
                self._maybe_save(actor_step, env_steps)

                if with_summaries:
                    summary = result[-1]
                    self.summary_writer.add_summary(summary, global_step=env_steps)

                sample_kl = result[0]
                kl_running_avg = (kl_running_avg + sample_kl) / 2  # running avg with exponential weights for past

                if kl_running_avg > self.params.target_kl:
                    log.info(
                        'Early stopping after %d/%d steps because of high KL divergence %f > %f',
                        epoch + 1, self.params.ppo_epochs, sample_kl, self.params.target_kl,
                    )
                    early_stop = True
                    break

            if early_stop:
                log.info('Early stopping after %d of %d epochs...', epoch + 1, self.params.ppo_epochs)
                break

        return actor_step

    def _train_critic(self, buffer, env_steps, objectives, actor_critic, train_critic, critic_step, critic_summaries):
        # train critic
        summary = None
        critic_step = critic_step.eval(session=self.session)

        if len(buffer) <= 0:
            return

        prev_loss = 1e10
        for epoch in range(self.params.ppo_epochs):
            losses = []
            buffer.shuffle()

            for i in range(0, len(buffer), self.params.batch_size):
                with_summaries = self._should_write_summaries(critic_step) and summary is None
                summaries = [critic_summaries] if with_summaries else []

                start, end = i, i + self.params.batch_size

                policy_input = actor_critic.input_dict(
                    buffer.obs[start:end], buffer.goals[start:end],
                    buffer.neighbors[start:end], buffer.num_neighbors[start:end],
                )

                result = self.session.run(
                    [objectives.critic_loss, train_critic] + summaries,
                    feed_dict={
                        self.ph_returns: buffer.returns[start:end],
                        self.ph_masks: buffer.masks[start:end],
                        **policy_input
                    },
                )

                critic_step += 1
                losses.append(result[0])

                if with_summaries:
                    summary = result[-1]
                    self.summary_writer.add_summary(summary, global_step=env_steps)

            # check loss improvement at the end of each epoch, early stop if necessary
            avg_loss = np.mean(losses)
            if avg_loss >= prev_loss:
                log.info('Early stopping after %d epochs because critic did not improve', epoch + 1)
                log.info('Was %.4f now %.4f, ratio %.3f', prev_loss, avg_loss, avg_loss / prev_loss)
                break
            prev_loss = avg_loss

    def _is_bootstrap(self, env_steps):
        """Check whether we're still in the initial bootstrapping stage."""
        return env_steps < self.params.bootstrap_env_steps

    def _maybe_train_reachability(self, data, env_steps):
        if not data.has_enough_data():
            return

        buffer = data.buffer

        batch_size = self.params.reachability_batch_size
        summary = None
        reach_step = self.reach_step.eval(session=self.session)

        prev_loss = 1e10
        num_epochs = self.params.reachability_train_epochs
        if self._is_bootstrap(env_steps):
            num_epochs = max(5, num_epochs * 2)  # during bootstrap do more epochs to train faster!

        log.info('Training reachability %d pairs, batch %d, epochs %d', len(buffer), batch_size, num_epochs)

        for epoch in range(num_epochs):
            losses = []
            buffer.shuffle_data()
            obs_first, obs_second, labels = buffer.obs_first, buffer.obs_second, buffer.labels

            for i in range(0, len(obs_first) - 1, batch_size):
                with_summaries = self._should_write_summaries(reach_step) and summary is None
                summaries = [self.reach_summaries] if with_summaries else []

                start, end = i, i + batch_size

                result = self.session.run(
                    [self.reachability.loss, self.train_reachability] + summaries,
                    feed_dict={
                        self.reachability.ph_obs_first: obs_first[start:end],
                        self.reachability.ph_obs_second: obs_second[start:end],
                        self.reachability.ph_labels: labels[start:end],
                    }
                )

                reach_step += 1
                self._maybe_save(reach_step, env_steps)
                losses.append(result[0])

                if with_summaries:
                    summary = result[-1]
                    self.summary_writer.add_summary(summary, global_step=env_steps)

            # check loss improvement at the end of each epoch, early stop if necessary
            avg_loss = np.mean(losses)
            if avg_loss >= prev_loss:
                log.info('Early stopping after %d epochs because reachability did not improve', epoch + 1)
                log.info('Was %.4f now %.4f, ratio %.3f', prev_loss, avg_loss, avg_loss / prev_loss)
                break
            prev_loss = avg_loss

    def _maybe_train_locomotion(self, data, env_steps):
        if not data.has_enough_data():
            return

        batch_size = self.params.locomotion_batch_size
        summary = None
        loco_step = self.locomotion_step.eval(session=self.session)

        prev_loss = 1e10

        num_epochs = self.params.locomotion_train_epochs
        log.info('Training locomotion %d pairs, batch %d, epochs %d', len(data.buffer), batch_size, num_epochs)

        for epoch in range(num_epochs):
            losses = []
            data.shuffle_data()
            obs_curr, obs_goal, actions = data.buffer.obs_curr, data.buffer.obs_goal, data.buffer.actions

            for i in range(0, len(obs_curr) - 1, batch_size):
                with_summaries = self._should_write_summaries(loco_step) and summary is None
                summaries = [self.locomotion_summaries] if with_summaries else []

                start, end = i, i + batch_size

                result = self.session.run(
                    [self.locomotion.loss, self.train_locomotion] + summaries,
                    feed_dict={
                        self.locomotion.ph_obs_curr: obs_curr[start:end],
                        self.locomotion.ph_obs_goal: obs_goal[start:end],
                        self.locomotion.ph_actions: actions[start:end],
                    }
                )

                loco_step += 1
                losses.append(result[0])

                if with_summaries:
                    summary = result[-1]
                    self.summary_writer.add_summary(summary, global_step=env_steps)

            # check loss improvement at the end of each epoch, early stop if necessary
            avg_loss = np.mean(losses)
            if avg_loss >= prev_loss:
                log.info('Early stopping after %d epochs because locomotion did not improve', epoch)
                log.info('Was %.4f now %.4f, ratio %.3f', prev_loss, avg_loss, avg_loss / prev_loss)
                break
            prev_loss = avg_loss

    def _learn_loop(self, multi_env):
        """Main training loop."""
        step, env_steps = self.session.run([self.actor_step, self.total_env_steps])

        env_obs = multi_env.reset()
        infos = multi_env.info()
        observations, goals = main_observation(env_obs), goal_observation(env_obs)

        buffer = TmaxPPOBuffer()

        trajectory_buffer = TrajectoryBuffer(multi_env.num_envs)  # separate buffer for complete episode trajectories
        reachability_buffer = ReachabilityBuffer(self.params)
        locomotion_buffer = LocomotionBuffer(self.params)

        tmax_mgr = self.tmax_mgr
        tmax_mgr.initialize(observations, infos)
        bonuses = [0] * multi_env.num_envs

        def end_of_training(s, es):
            return s >= self.params.train_for_steps or es > self.params.train_for_env_steps

        while not end_of_training(step, env_steps):
            # collecting experience
            timing = Timing()
            num_steps = 0
            batch_start = time.time()
            with timing.timeit('experience'):
                buffer.reset()
                is_bootstrap = self._is_bootstrap(env_steps)
                for rollout_step in range(self.params.rollout):
                    neighbors, num_neigh = self.tmax_mgr.get_neighbors()
                    actions, action_probs, values, masks = self.policy_step(
                        observations, goals, neighbors, num_neigh, is_bootstrap,
                    )

                    # wait for all the workers to complete an environment step
                    env_obs, rewards, dones, infos = multi_env.step(actions)
                    new_observations, new_goals = main_observation(env_obs), goal_observation(env_obs)

                    trajectory_buffer.add(observations, actions, dones, tmax_mgr)
                    bonuses, loco_rewards, done_flags = tmax_mgr.update(
                        new_observations, new_goals, dones, infos, is_bootstrap,
                    )
                    rewards, dones = tmax_mgr.combine_rewards_and_dones(
                        rewards, dones, bonuses, loco_rewards, done_flags,
                    )
                    # add experience from all environments to the current buffer(s)
                    buffer.add(
                        observations, goals, actions, action_probs,
                        rewards, dones, values,
                        neighbors, num_neigh, masks,
                    )
                    observations, goals = new_observations, new_goals

                    num_steps += num_env_steps(infos)

                # last step values are required for TD-return calculation
                neighbors, num_neigh = tmax_mgr.get_neighbors()
                _, _, values, _ = self.policy_step(observations, goals, neighbors, num_neigh, is_bootstrap)
                buffer.values.append(values)

            env_steps += num_steps

            # calculate discounted returns and GAE
            buffer.finalize_batch(self.params.gamma, self.params.gae_lambda)

            if self.params.use_locomotion:
                loco_buffer, buffer = buffer.split_by_mask()

            traj_len, traj_nbytes = trajectory_buffer.obs_size()
            log.info('Len trajectories %d size %.1fM', traj_len, traj_nbytes / 1e6)

            # update actor and critic
            with timing.timeit('train'):
                if not is_bootstrap:
                    step = self._train_actor(
                        buffer, env_steps,
                        self.objectives, self.actor_critic, self.train_actor, self.actor_step, self.actor_summaries,
                    )
                    self._train_critic(
                        buffer, env_steps,
                        self.objectives, self.actor_critic, self.train_critic, self.critic_step, self.critic_summaries,
                    )

            # update reachability net
            with timing.timeit('reach'):
                reachability_buffer.extract_data(trajectory_buffer.complete_trajectories)
                self._maybe_train_reachability(reachability_buffer, env_steps)

            # update locomotion net
            if self.params.use_locomotion:
                with timing.timeit('locomotion'):
                    if self.params.rl_locomotion:
                        if not is_bootstrap:
                            self._train_actor(
                                loco_buffer, env_steps,
                                self.loco_objectives, self.loco_actor_critic, self.train_loco_actor,
                                self.loco_actor_step, self.loco_actor_summaries,
                            )
                            self._train_critic(
                                loco_buffer, env_steps,
                                self.loco_objectives, self.loco_actor_critic, self.train_loco_critic,
                                self.loco_critic_step, self.loco_critic_summaries,
                            )
                    else:
                        locomotion_buffer.extract_data(trajectory_buffer.complete_trajectories)
                        self._maybe_train_locomotion(locomotion_buffer, env_steps)

            avg_reward = multi_env.calc_avg_rewards(n=self.params.stats_episodes)
            avg_length = multi_env.calc_avg_episode_lengths(n=self.params.stats_episodes)
            fps = num_steps / (time.time() - batch_start)

            self._maybe_print(step, env_steps, avg_reward, avg_length, fps, timing)
            self._maybe_aux_summaries(tmax_mgr, env_steps, avg_reward, avg_length, fps, bonuses)
            self._maybe_tmax_summaries(tmax_mgr, env_steps)
            self._maybe_update_avg_reward(avg_reward, multi_env.stats_num_episodes())
            self._maybe_trajectory_summaries(trajectory_buffer, env_steps)

            trajectory_buffer.reset_trajectories()
            # encoder changed, so we need to re-encode all landmarks
            tmax_mgr.landmarks_encoder.reset()

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
