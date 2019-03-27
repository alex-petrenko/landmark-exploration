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
from algorithms.baselines.ppo.agent_ppo import PPOBuffer, AgentPPO
from algorithms.curiosity.reachability_curiosity.observation_encoder import ObservationEncoder
from algorithms.curiosity.reachability_curiosity.reachability_curiosity import ReachabilityCuriosityModule
from algorithms.encoders import make_encoder, make_encoder_with_goal
from algorithms.env_wrappers import main_observation_space, is_goal_based_env
from algorithms.models import make_model
from algorithms.multi_env import MultiEnv
from algorithms.tf_utils import dense, count_total_parameters, placeholder_from_space, placeholders, \
    image_summaries_rgb, summary_avg_min_max, merge_summaries, tf_shape, placeholder
from algorithms.tmax.graph_encoders import make_graph_encoder
from algorithms.tmax.locomotion import LocomotionNetwork
from algorithms.tmax.tmax_utils import TmaxMode, TmaxTrajectoryBuffer, TmaxReachabilityBuffer
from algorithms.topological_maps.localization import Localizer
from algorithms.topological_maps.topological_map import TopologicalMap, map_summaries, hash_observation
from utils.distributions import CategoricalProbabilityDistribution
from utils.timing import Timing
from utils.utils import log, AttrDict, numpy_all_the_way


class ActorCritic:
    def __init__(self, env, ph_observations, params, has_goal, name):
        with tf.variable_scope(name):
            obs_space = main_observation_space(env)

            self.ph_observations = ph_observations

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
            act_encoder = tf.make_template(
                'act_obs_enc', make_encoder_func, create_scope_now_=True,
                obs_space=obs_space, regularizer=reg, params=params,
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

            value_model = make_model(value_obs_and_neighborhoods, reg, params, 'val_mdl')

            value_fc = dense(value_model.latent, params.model_fc_size // 2, reg)
            self.value = tf.squeeze(tf.contrib.layers.fully_connected(value_fc, 1, activation_fn=None), axis=[1])

            log.info('Total parameters so far: %d', count_total_parameters())

    def input_dict(self, observations, goals, neighbors_encoded, num_neighbors):
        feed_dict = {self.ph_observations: observations}
        if self.has_goal:
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
        """This is mainly used to precalculate the landmark embeddings for graph encoder."""
        return session.run(self.encode_single_obs, feed_dict={self.ph_observations: landmarks})


class TmaxPPOBuffer(PPOBuffer):
    def __init__(self):
        super(TmaxPPOBuffer, self).__init__()
        self.neighbors, self.num_neighbors = None, None
        self.modes = None
        self.masks = None

    def reset(self):
        super(TmaxPPOBuffer, self).reset()
        self.neighbors, self.num_neighbors = [], []
        self.modes = []
        self.masks = []

    # noinspection PyMethodOverriding
    def add(self, obs, goals, actions, action_probs, rewards, dones, values, neighbors, num_neighbors, modes, masks):
        """Append one-step data to the current batch of observations."""
        args = copy.copy(locals())
        super(TmaxPPOBuffer, self)._add_args(args)

    def split_by_mode(self):
        buffers = {}
        for mode in TmaxMode.all_modes():
            buffers[mode] = TmaxPPOBuffer()
            buffers[mode].reset()

        for i in range(len(self)):
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
        self.neighbors_buffer = np.zeros([
            self.params.num_envs, self.params.max_neighborhood_size, agent.encoded_landmark_size,
        ])
        self.current_maps = None
        # we need to potentially preserve a few most recent copies of the persistent map
        # because when we update the persistent map not all of the environments switch to it right away,
        # we might need to wait until the episode end in all of them
        self.persistent_maps = deque([])

        self.map_size_before_exploration = self.map_size_before_locomotion = 0

        self.landmarks_encoder = ObservationEncoder(agent.actor_critic.encode_landmarks)

        self.episode_frames = [0] * self.num_envs
        self.episode_locomotion_reward = [0] * self.num_envs

        self.locomotion_prev = [None] * self.num_envs  # starting landmark for locomotion policy
        self.locomotion_targets = [None] * self.num_envs  # immediate goal for locomotion policy
        self.locomotion_final_targets = [None] * self.num_envs  # final target (e.g. goal observation)
        self.last_locomotion_success = [0] * self.num_envs
        self.locomotion_achieved_goal = deque([])
        self.locomotion_traversal_length = deque([])

        self.global_stage = TmaxMode.EXPLORATION
        self.last_stage_change = self.params.reachability_bootstrap

        self.mode = [TmaxMode.EXPLORATION] * self.num_envs
        self.env_stage = [TmaxMode.EXPLORATION] * self.num_envs

        self.idle_frames = [0] * self.num_envs
        self.action_frames = [np.random.randint(1, 3) for _ in range(self.num_envs)]

        self.deliberate_action = [True] * self.num_envs

        self.localizer = Localizer(self.params, self.curiosity.obs_encoder)

    def initialize(self, obs, info, env_steps):
        if self.initialized:
            return

        self.current_maps = []
        for i in range(self.num_envs):
            self.current_maps.append(TopologicalMap(obs[i], directed_graph=False, initial_info=info[i]))

        self.persistent_maps.append(TopologicalMap(obs[0], directed_graph=False, initial_info=info[0]))

        self.last_stage_change = max(self.last_stage_change, env_steps)

        self.initialized = True
        return self.mode

    def _log_verbose(self, s, *args):
        if self._verbose:
            log.debug(s, *args)

    def get_locomotion_targets(self, env_indices):
        assert len(env_indices) > 0
        targets, target_hashes = [], []

        for env_i in env_indices:
            assert self.mode[env_i] == TmaxMode.LOCOMOTION
            locomotion_target_idx = self.locomotion_targets[env_i]
            assert locomotion_target_idx is not None
            target_obs = self.current_maps[env_i].get_observation(locomotion_target_idx)
            targets.append(target_obs)
            target_hashes.append(self.current_maps[env_i].get_hash(locomotion_target_idx))

        assert len(targets) == len(env_indices)
        return targets, target_hashes

    def _select_next_locomotion_target(self, env_i, curr_landmark_idx, verbose=False):
        assert self.mode[env_i] == TmaxMode.LOCOMOTION

        final_target = self.locomotion_final_targets[env_i]
        assert final_target is not None

        m = self.current_maps[env_i]
        path = m.get_path(curr_landmark_idx, final_target)
        assert path is not None  # final goal should always be reachable!

        if verbose and len(path) > 1:
            log.info('Shortest path from %d to %d is %r', curr_landmark_idx, final_target, path)

        if curr_landmark_idx == final_target or len(path) <= 1:
            # we reached the final target
            if self.env_stage[env_i] == TmaxMode.EXPLORATION:
                # reached the target, switch to exploration policy
                self.mode[env_i] = TmaxMode.EXPLORATION
                self.locomotion_prev[env_i] = None
                self.locomotion_targets[env_i] = None
                self.locomotion_final_targets[env_i] = None
            else:
                # sample new "final goal" for locomotion policy
                locomotion_goal_idx = self._get_locomotion_final_goal(env_i, curr_landmark_idx)
                self.locomotion_final_targets[env_i] = locomotion_goal_idx
                # call recursively to re-set the locomotion target
                self._select_next_locomotion_target(env_i, curr_landmark_idx)
        else:
            # set the next landmark on the way to the final goal to be our next target
            assert len(path) >= 2
            self.locomotion_prev[env_i] = curr_landmark_idx
            self.locomotion_targets[env_i] = path[1]  # next vertex on the path

    def _accessible_targets(self, env_i, curr_landmark_idx):
        m = self.current_maps[env_i]
        reliable_path_length = -math.log(self.params.reliable_path_probability)
        path_lengths_to_targets = m.path_lengths(curr_landmark_idx)

        accessible_targets = []
        for target_idx, path_length in path_lengths_to_targets.items():
            if path_length < reliable_path_length:
                accessible_targets.append(target_idx)

        return accessible_targets

    def _get_locomotion_final_goal(self, env_i, curr_landmark_idx):
        m = self.current_maps[env_i]
        reachable_indices = m.reachable_indices(curr_landmark_idx)

        if self.env_stage[env_i] == TmaxMode.LOCOMOTION:
            # sample anything except starting landmark
            reachable_indices.remove(curr_landmark_idx)
            assert len(reachable_indices) > 0  # should always be true because we check during stage change

            # randomly sample landmark from the persistent map
            locomotion_goal_idx = random.choice(reachable_indices)
        else:
            accessible_targets = self._accessible_targets(env_i, curr_landmark_idx)
            # current vertex has path length of 0 to itself and is always accessible
            assert len(accessible_targets) > 0

            # TODO: instead of randomly picking target, choose the most "interesting" one
            locomotion_goal_idx = random.choice(accessible_targets)

        return locomotion_goal_idx

    def _prepare_persistent_map_for_locomotion(self, env_steps):
        """
        Keep only vertices topologically close to starting location (progressively grow the map).
        This is kind of natural curriculum, to avoid making problem too complex from the very beginning.
        TODO: instead, allow map to grow 1 edge at a time from the previous map
        """
        # select the map with most landmarks
        locomotion_map_idx = 0
        maps = self.current_maps

        for env_i in range(self.num_envs):
            if maps[env_i].num_landmarks() > maps[locomotion_map_idx].num_landmarks():
                locomotion_map_idx = env_i

        log.debug(
            'Map %d is selected as a locomotion map, with %d landmarks',
            locomotion_map_idx, maps[locomotion_map_idx].num_landmarks(),
        )

        # TODO: remove all the localization variables from the map and move them to localizer
        m = copy.deepcopy(maps[locomotion_map_idx])
        topological_distances = m.topological_distances(from_idx=0)  # 0 = first landmark

        remove_nodes = []

        locomotion_stage_number = env_steps // (2 * self.params.stage_duration)
        locomotion_stage_number = max(1, locomotion_stage_number)

        max_dist = locomotion_stage_number

        for node, distance in topological_distances.items():
            if distance > max_dist:
                remove_nodes.append(node)

        m.graph.remove_nodes_from(remove_nodes)

        m.new_episode()

        topological_distances = m.topological_distances(from_idx=0)
        assert all(d <= max_dist for d in topological_distances.values())

        self.map_size_before_locomotion = m.num_landmarks()
        self.persistent_maps.append(m)
        log.debug(
            'Pruned map %d, nodes %r, distances %r, num persistent maps %d',
            locomotion_map_idx, m.graph.nodes, m.topological_distances(from_idx=0), len(self.persistent_maps),
        )

    def _prepare_persistent_map_for_exploration(self):
        """Keep only edges with high probability of success, delete inaccessible vertices."""
        m = copy.deepcopy(self.persistent_maps[-1])  # copy newest persistent map

        # remove individual edges with low probability
        remove_edges = set()  # unique edges, to avoid deleting the same edge twice
        g = m.graph
        for e in g.edges():
            i1, i2 = e
            unreliable_edge = g[i1][i2]['success'] < self.params.reliable_edge_probability
            unreliable_edge_back = g[i2][i1]['success'] < self.params.reliable_edge_probability
            unreliable_edge = unreliable_edge and unreliable_edge_back

            edge_too_short = g[i1][i2]['last_traversal_frames'] < self.params.reachable_threshold
            edge_too_short_back = g[i2][i1]['last_traversal_frames'] < self.params.reachable_threshold
            edge_too_short = edge_too_short and edge_too_short_back

            # remove edge if it's useless in both directions
            if unreliable_edge or edge_too_short:
                remove_edges.add(e)

        m.remove_edges_from(list(remove_edges))

        # remove all isolated (non-reachable) vertices
        reachable_targets = m.reachable_indices(0)
        remove_vertices = []
        for target_idx in m.graph.nodes():
            if target_idx not in reachable_targets:
                remove_vertices.append(target_idx)

        assert len(remove_vertices) < len(m.graph.nodes)
        m.graph.remove_nodes_from(remove_vertices)

        m.new_episode()

        self.map_size_before_exploration = m.num_landmarks()
        self.persistent_maps.append(m)
        log.info(
            'Prune map for exploration, vertices %d, remove vertices %r, num persistent maps %d',
            m.num_landmarks(), remove_vertices, len(self.persistent_maps),
        )

    def _reset_episodic_map(self, env_i):
        self.curiosity.episodic_maps[env_i] = copy.deepcopy(self.persistent_maps[-1])
        self.curiosity.episodic_maps[env_i].new_episode()

    def _new_episode(self, env_i):
        if self.global_stage == TmaxMode.LOCOMOTION:
            self.current_maps[env_i] = self.persistent_maps[-1]  # use literally the same map instance in all envs
        else:
            if self.global_stage != self.env_stage[env_i]:
                # update map!
                self.current_maps[env_i] = copy.deepcopy(self.persistent_maps[-1])

        self.env_stage[env_i] = self.global_stage

        self.current_maps[env_i].new_episode()

        # delete old persistent maps that aren't used anymore
        while len(self.persistent_maps) > 1:
            used_by_env = False
            for i in range(self.num_envs):
                if self.current_maps[i] is self.persistent_maps[0]:
                    used_by_env = True
            if not used_by_env:
                log.debug(
                    'Delete old persistent map with %d landmarks, it is not used anymore!',
                    self.persistent_maps[0].num_landmarks(),
                )
                self.persistent_maps.popleft()
            else:
                break

        self._reset_episodic_map(env_i)

        self.episode_frames[env_i] = self.last_locomotion_success[env_i] = 0
        self.episode_locomotion_reward[env_i] = 0

        self.locomotion_prev[env_i] = None
        self.locomotion_targets[env_i] = self.locomotion_final_targets[env_i] = None

        # only for montezuma
        self.idle_frames[env_i] = 0
        self.action_frames[env_i] = np.random.randint(1, 3)

        curr_landmark_idx = 0
        locomotion_goal_idx = self._get_locomotion_final_goal(env_i, curr_landmark_idx)
        if self.env_stage[env_i] == TmaxMode.LOCOMOTION:
            assert locomotion_goal_idx != curr_landmark_idx

        self.mode[env_i] = TmaxMode.LOCOMOTION
        self.locomotion_final_targets[env_i] = locomotion_goal_idx
        self._select_next_locomotion_target(env_i, curr_landmark_idx, verbose=True)

        assert self.mode[env_i] is not None
        if self.mode[env_i] == TmaxMode.LOCOMOTION:
            assert self.locomotion_prev[env_i] == curr_landmark_idx
            assert self.locomotion_targets[env_i] is not None
            assert self.locomotion_final_targets[env_i] is not None

    def _update_stage(self, env_steps):
        stage_changed = False
        if env_steps - self.last_stage_change > self.params.stage_duration:
            if self.global_stage == TmaxMode.EXPLORATION:
                # it's time to switch to locomotion stage, but only if all the persistent maps are not empty
                # (so we can have some locomotion targets)
                has_non_empty_map = False
                for env_i in range(self.num_envs):
                    if len(self.current_maps[env_i].reachable_indices(0)) > 1:
                        has_non_empty_map = True
                        break

                if has_non_empty_map:
                    # all the maps are non-empty, we can change to locomotion
                    self.global_stage = TmaxMode.LOCOMOTION
                    self._prepare_persistent_map_for_locomotion(env_steps)
                    stage_changed = True
                    log.debug('Stage changed to Locomotion')
            else:
                self.global_stage = TmaxMode.EXPLORATION
                self._prepare_persistent_map_for_exploration()
                stage_changed = True
                log.debug('Stage changed to Exploration')

        if stage_changed:
            self.last_stage_change = env_steps

    def _update_exploration(self, next_obs, infos):
        """Expand the persistent maps in exploration mode."""
        if not self.curiosity.is_initialized():
            # don't update persistent maps when distance metric is not trained
            return

        # call localize only for environments in exploration mode
        maps = [None] * self.num_envs
        for env_i in range(self.num_envs):
            if self.mode[env_i] == TmaxMode.EXPLORATION:
                maps[env_i] = self.current_maps[env_i]

        self.localizer.localize(
            self.agent.session, next_obs, infos, maps, self.curiosity.reachability,
        )

    def _distance_to_locomotion_targets(self, obs):
        loco_env_indices = []
        loco_curr_obs = []
        for env_i in range(self.num_envs):
            if self.mode[env_i] == TmaxMode.LOCOMOTION:
                loco_env_indices.append(env_i)
                loco_curr_obs.append(obs[env_i])

        if len(loco_env_indices) == 0:
            return []

        # TODO: integrate obs encoder into reachability?
        obs_hashes = [hash_observation(o) for o in loco_curr_obs]
        self.curiosity.obs_encoder.encode(self.agent.session, loco_curr_obs, obs_hashes)
        loco_curr_obs_encoded = [self.curiosity.obs_encoder.encoded_obs[obs_hash] for obs_hash in obs_hashes]

        locomotion_targets, target_hashes = self.get_locomotion_targets(loco_env_indices)
        self.curiosity.obs_encoder.encode(self.agent.session, locomotion_targets, target_hashes)
        targets_encoded = [self.curiosity.obs_encoder.encoded_obs[h] for h in target_hashes]

        distances = self.curiosity.reachability.distances(self.agent.session, loco_curr_obs_encoded, targets_encoded)

        all_distances = [None] * self.num_envs
        for i, env_i in enumerate(loco_env_indices):
            all_distances[env_i] = distances[i]
        return all_distances

    def _update_locomotion(self, next_obs, dones):
        rewards = np.zeros(self.num_envs)
        locomotion_dones = np.zeros(self.num_envs, dtype=bool)

        if not self.curiosity.is_initialized():
            # don't update anything when distance metric is not trained
            return rewards, locomotion_dones

        distances = self._distance_to_locomotion_targets(next_obs)
        successful_traversal_frames = self.params.successful_traversal_frames

        for env_i in range(self.num_envs):
            if self.mode[env_i] != TmaxMode.LOCOMOTION:
                continue

            since_last_success = self.episode_frames[env_i] - self.last_locomotion_success[env_i]

            episode_ended = dones[env_i]
            locomotion_timed_out = \
                self.env_stage[env_i] == TmaxMode.EXPLORATION \
                and since_last_success > successful_traversal_frames

            locomotion_failed = episode_ended or locomotion_timed_out

            if locomotion_failed:
                # locomotion not successful
                locomotion_dones[env_i] = True

                self.current_maps[env_i].update_edge_traversal(
                    self.locomotion_prev[env_i], self.locomotion_targets[env_i], 0, frames=math.inf,
                )
                self.locomotion_achieved_goal.append(0)
                rewards[env_i] -= 5.0  # failed locomotion, to prevent being attracted by the goal

                log.info(
                    'Locomotion failed dist %.3f, prev %d, goal %d, frame %d, since last success %d',
                    distances[env_i], self.locomotion_prev[env_i],
                    self.locomotion_targets[env_i], self.episode_frames[env_i], since_last_success,
                )

                if locomotion_timed_out:
                    self.mode[env_i] = TmaxMode.EXPLORATION
                    self.locomotion_prev[env_i] = None
                    self.locomotion_targets[env_i] = None
                    self.locomotion_final_targets[env_i] = None

                continue

            assert distances[env_i] is not None and 0.0 <= distances[env_i] <= 1.0

            rewards[env_i] -= distances[env_i] * 0.05  # scaling factor for dense reward

            if distances[env_i] < self.params.locomotion_reached_threshold:  # TODO: low-pass filter?
                # reached the target!
                curr_landmark_idx = self.locomotion_targets[env_i]

                rewards[env_i] += 1

                log.info(
                    'Locomotion net gets reward +1! (%.3f) dist %.3f, prev %d, goal %d, frame %d, took %d env %d',
                    rewards[env_i], distances[env_i], self.locomotion_prev[env_i],
                    curr_landmark_idx, self.episode_frames[env_i], since_last_success, env_i,
                )

                locomotion_dones[env_i] = True
                is_success = 1 if since_last_success < successful_traversal_frames else 0
                self.current_maps[env_i].update_edge_traversal(
                    self.locomotion_prev[env_i], self.locomotion_targets[env_i], is_success, frames=since_last_success,
                )
                self.last_locomotion_success[env_i] = self.episode_frames[env_i]
                self.locomotion_achieved_goal.append(1)
                self.locomotion_traversal_length.append(since_last_success)

                self._select_next_locomotion_target(env_i, curr_landmark_idx)
                if self.mode[env_i] != TmaxMode.LOCOMOTION:
                    log.info('Reached the locomotion goal and switched to exploration')

        return rewards, locomotion_dones

    def update(self, obs, next_obs, rewards, dones, infos, env_steps, timing=None, verbose=False):
        self._verbose = verbose
        if timing is None:
            timing = Timing()

        assert len(obs) == len(self.current_maps)

        with timing.add_time('curiosity'):
            curiosity_bonus = self.curiosity.generate_bonus_rewards(
                self.agent.session, obs, next_obs, None, dones, infos,
            )

        with timing.add_time('update_exploration'):
            self._update_exploration(next_obs, infos)

        with timing.add_time('update_locomotion'):
            locomotion_rewards, locomotion_dones = self._update_locomotion(next_obs, dones)

        for env_i in range(self.num_envs):
            if dones[env_i]:
                self._new_episode(env_i)
            else:
                self.episode_frames[env_i] += 1

        self._update_stage(env_steps)

        augmented_rewards = np.zeros(self.num_envs)
        done_flags = np.zeros(self.num_envs, dtype=bool)
        modes = np.array(self.mode, np.int32)

        # combine final rewards and done flags
        for env_i in range(self.num_envs):
            if dones[env_i]:
                continue

            if self.mode[env_i] == TmaxMode.EXPLORATION:
                augmented_rewards[env_i] = rewards[env_i]
                if not dones[env_i]:
                    augmented_rewards[env_i] += curiosity_bonus[env_i]
                done_flags[env_i] = dones[env_i]
            else:
                augmented_rewards[env_i] = locomotion_rewards[env_i]
                self.episode_locomotion_reward[env_i] += locomotion_rewards[env_i]
                done_flags[env_i] = dones[env_i] or locomotion_dones[env_i]

        return augmented_rewards, done_flags, modes

    # not used, may contain errors
    # noinspection PyUnresolvedReferences
    def get_neighbors(self):
        """Not used now."""
        if not self.params.use_neighborhood_encoder:
            return None, None

        neighbors_buffer = self.neighbors_buffer
        maps = self.current_maps

        neighbors_buffer.fill(0)
        num_neighbors = [0] * len(maps)
        landmark_env_idx = []

        neighbor_landmarks, neighbor_hashes = [], []

        for env_idx, m in enumerate(maps):
            n_indices = m.neighborhood()
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

                neighbor_landmarks.append(m.get_observation(n_idx))
                neighbor_hashes.append(m.get_hash(n_idx))
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

    class Params(
        AgentPPO.Params,
        ReachabilityCuriosityModule.Params,
    ):
        """Hyperparams for the algorithm and the training process."""

        def __init__(self, experiment_name):
            """Default parameter values set in ctor."""
            # calling all parent constructors
            AgentPPO.Params.__init__(self, experiment_name)
            ReachabilityCuriosityModule.Params.__init__(self)

            # TMAX-specific parameters
            self.use_neighborhood_encoder = False
            self.graph_enc_name = 'rnn'  # 'rnn', 'deepsets'
            self.max_neighborhood_size = 6  # max number of neighbors that can be fed into policy at every timestep
            self.graph_encoder_rnn_size = 128  # size of GRU layer in RNN neighborhood encoder

            self.locomotion_max_trajectory = 35  # max trajectory length to be utilized for locomotion training
            self.locomotion_target_buffer_size = 100000  # target number of (obs, goal, action) tuples to store
            self.locomotion_train_epochs = 1
            self.locomotion_batch_size = 256
            self.rl_locomotion = True
            self.locomotion_reached_threshold = 0.1  # if distance is less than that, we reached the target
            self.reliable_path_probability = 0.5  # product of probs along the path for it to be considered reliable
            self.reliable_edge_probability = 0.8
            self.successful_traversal_frames = 40  # if we traverse an edge in less than that, we succeeded

            self.stage_duration = 1000000

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

        self.actor_critic = ActorCritic(
            env, self.ph_observations, self.params, has_goal=is_goal_based_env(env), name='main',
        )

        if self.params.rl_locomotion:
            self.loco_actor_critic = ActorCritic(env, self.ph_observations, self.params, has_goal=True, name='loco')
        else:
            self.locomotion = LocomotionNetwork(env, params)

        if self.params.use_neighborhood_encoder is None:
            self.encoded_landmark_size = 1
        else:
            self.encoded_landmark_size = self.actor_critic.encoded_obs_size

        self.curiosity = ReachabilityCuriosityModule(env, params)
        self.curiosity.reachability_buffer = TmaxReachabilityBuffer(params)

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
        else:
            locomotion_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='locomotion_opt')
            self.train_locomotion = locomotion_opt.minimize(self.locomotion.loss, global_step=self.locomotion_step)

        # summaries
        self.add_summaries()

        self.actor_summaries = merge_summaries(collections=['actor'])
        self.critic_summaries = merge_summaries(collections=['critic'])

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
                image_summaries_rgb(self.actor_critic.ph_goal_obs, name='goal', collections=['actor'])

        self.add_ppo_summaries(self.actor_critic, self.objectives, self.actor_step, self.critic_step)

        if self.params.rl_locomotion:
            image_summaries_rgb(self.loco_actor_critic.ph_goal_obs, name='loco_goal', collections=['loco_actor'])

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

        log.info('Avg. %d episode length: %.3f', self.params.stats_episodes, avg_length)
        best_avg_reward = self.best_avg_reward.eval(session=self.session)
        log.info(
            'Avg. %d episode reward: %.3f (best: %.3f)',
            self.params.stats_episodes, avg_rewards, best_avg_reward,
        )

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
        maps = tmax_mgr.current_maps
        map_summaries(maps, env_steps, self.summary_writer, 'tmax_maps')

        map_summaries([tmax_mgr.persistent_maps[-1]], env_steps, self.summary_writer, 'tmax_persistent_map')

        # locomotion summaries
        stats_samples = 300

        summary_obj = tf.Summary()
        achieved_goal = tmax_mgr.locomotion_achieved_goal
        if len(achieved_goal) >= 10:
            while len(achieved_goal) > stats_samples:
                achieved_goal.popleft()
            summary_obj.value.add(tag='locomotion/avg_success', simple_value=np.mean(achieved_goal))

        traversal_len = tmax_mgr.locomotion_traversal_length
        if len(traversal_len) >= 10:
            while len(traversal_len) > stats_samples:
                traversal_len.popleft()
            summary_obj.value.add(tag='locomotion/avg_traversal_len', simple_value=np.mean(traversal_len))

        traversal_success = []
        for e, properties in tmax_mgr.persistent_maps[-1].graph.edges.items():
            if properties['traversed']:
                traversal_success.append(properties['success'])
        if len(traversal_success) > 0:
            summary_obj.value.add(
                tag='locomotion/edge_traversal_success', simple_value=np.mean(traversal_success),
            )

        summary_obj.value.add(
            tag='locomotion/episode_locomotion_reward', simple_value=np.mean(tmax_mgr.episode_locomotion_reward),
        )
        summary_obj.value.add(
            tag='tmax_maps/map_size_before_locomotion',
            simple_value=np.mean(tmax_mgr.map_size_before_locomotion),
        )
        summary_obj.value.add(
            tag='tmax_maps/map_size_before_exploration',
            simple_value=np.mean(tmax_mgr.map_size_before_exploration),
        )

        summary_obj.value.add(tag='tmax/global_stage', simple_value=tmax_mgr.global_stage)
        summary_obj.value.add(tag='tmax/avg_mode', simple_value=np.mean(tmax_mgr.mode))
        summary_obj.value.add(tag='tmax/avg_env_stage', simple_value=np.mean(tmax_mgr.env_stage))

        self.summary_writer.add_summary(summary_obj, env_steps)
        self.summary_writer.flush()

    def _maybe_trajectory_summaries(self, trajectory_buffer, env_steps):
        time_since_last = time.time() - self._last_trajectory_summary
        if time_since_last < self.params.gif_save_rate or not trajectory_buffer.complete_trajectories:
            return

        start_gif_summaries = time.time()

        self._last_trajectory_summary = time.time()
        num_envs = self.params.gif_summary_num_envs

        trajectories = []
        trajectories_locomotion = []

        for trajectory in trajectory_buffer.complete_trajectories[:num_envs]:
            if all(mode == TmaxMode.LOCOMOTION for mode in trajectory.mode):
                trajectories_locomotion.append(numpy_all_the_way(trajectory.obs)[:, :, :, -3:])
            else:
                trajectories.append(numpy_all_the_way(trajectory.obs)[:, :, :, -3:])

        if len(trajectories) > 0:
            self._write_gif_summaries(tag='obs_trajectories', gif_images=trajectories, step=env_steps)

        if len(trajectories_locomotion) > 0:
            self._write_gif_summaries(tag='loco_trajectories', gif_images=trajectories_locomotion, step=env_steps)

        log.info('Took %.3f seconds to write gif summaries', time.time() - start_gif_summaries)

    def best_action(self, observation):
        raise NotImplementedError('Use best_action_tmax instead')

    def best_action_tmax(self, observations, goals, deterministic=False):
        neighbors, num_neighbors = self.tmax_mgr.get_neighbors()
        actions = self.actor_critic.best_action(
            self.session, observations, goals, neighbors, num_neighbors, deterministic,
        )
        return actions[0]

    def _idle_explore_policy_step(
            self, env_i, observations, goals, neighbors, num_neighbors,
            actions, action_probs, values, masks, tmax_mgr):
        """This is only used for Montezuma."""
        if len(env_i) <= 0:
            # no envs use this now
            return

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
        if len(non_idle_i) > 0:
            neighbors_policy = num_neighbors_policy = None
            if neighbors is not None:
                neighbors_policy = neighbors[non_idle_i]
                num_neighbors_policy = num_neighbors[non_idle_i]
            actions[non_idle_i], action_probs[non_idle_i], values[non_idle_i] = self.actor_critic.invoke(
                self.session, observations[non_idle_i], goals[non_idle_i],
                neighbors_policy, num_neighbors_policy, deterministic=False,
            )

    def _locomotion_policy_step(self, env_i, observations, goals, actions, action_probs, values, masks, tmax_mgr):
        if len(env_i) <= 0:
            return

        goal_obs, hashes = tmax_mgr.get_locomotion_targets(env_i)
        goals[env_i] = goal_obs  # replace goals with locomotion goals

        assert len(goal_obs) == len(env_i)
        for env_index in env_i:
            tmax_mgr.deliberate_action[env_index] = True

        if self.params.rl_locomotion:
            actions[env_i], action_probs[env_i], values[env_i] = self.loco_actor_critic.invoke(
                self.session, observations[env_i], goals[env_i], None, None,
            )
            masks[env_i] = 1
        else:
            actions[env_i] = self.locomotion.navigate(
                self.session, observations[env_i], goals[env_i], deterministic=False,
            )
            masks[env_i] = 0

    def _exploration_policy_step(
            self, env_i, observations, goals, neighbors, num_neighbors,
            actions, action_probs, values, masks, tmax_mgr,
    ):
        if len(env_i) <= 0:
            return

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

    def policy_step(self, observations, goals, neighbors, num_neighbors):
        """Run exploration or locomotion policy depending on the state of the particular environment."""
        tmax_mgr = self.tmax_mgr
        num_envs = len(observations)

        env_indices = {TmaxMode.IDLE_EXPLORATION: [], TmaxMode.LOCOMOTION: [], TmaxMode.EXPLORATION: []}
        for env_i, mode in enumerate(tmax_mgr.mode):
            env_indices[mode].append(env_i)
        total_num_indices = sum(len(v) for v in env_indices.values())
        assert total_num_indices == num_envs

        observations = np.asarray(observations)
        goals = np.asarray(goals)
        if neighbors is not None:
            neighbors = np.asarray(neighbors)
            num_neighbors = np.asarray(num_neighbors)

        actions = np.empty(num_envs, np.int32)
        action_probs = np.ones(num_envs, np.float32)
        values = np.zeros(num_envs, np.float32)
        masks = np.zeros(num_envs, np.int32)

        # IDLE_EXPLORATION policy (explore + idle) is used only for Montezuma
        self._idle_explore_policy_step(
            env_indices[TmaxMode.IDLE_EXPLORATION], observations, goals, neighbors, num_neighbors,
            actions, action_probs, values, masks, tmax_mgr,
        )

        self._locomotion_policy_step(
            env_indices[TmaxMode.LOCOMOTION], observations, goals, actions, action_probs, values, masks, tmax_mgr,
        )

        self._exploration_policy_step(
            env_indices[TmaxMode.EXPLORATION], observations, goals, neighbors, num_neighbors,
            actions, action_probs, values, masks, tmax_mgr,
        )

        return actions, action_probs, values, masks, goals

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

    def _maybe_train_locomotion(self, data, env_steps):
        """Train locomotion using self-imitation."""
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

    def _train_tmax(self, step, buffer, env_steps, timing):
        buffers = buffer.split_by_mode()
        buffer.reset()  # discard the original data (before mode split)

        if self.curiosity.is_initialized():
            with timing.timeit('train_policy'):
                step = self._train_actor(
                    buffers[TmaxMode.EXPLORATION], env_steps,
                    self.objectives, self.actor_critic, self.train_actor, self.actor_step, self.actor_summaries,
                )
                self._train_critic(
                    buffers[TmaxMode.EXPLORATION], env_steps,
                    self.objectives, self.actor_critic, self.train_critic, self.critic_step, self.critic_summaries,
                )

            with timing.timeit('train_locomotion'):
                if self.params.rl_locomotion:
                    self._train_actor(
                        buffers[TmaxMode.LOCOMOTION], env_steps,
                        self.loco_objectives, self.loco_actor_critic, self.train_loco_actor,
                        self.loco_actor_step, self.loco_actor_summaries,
                    )
                    self._train_critic(
                        buffers[TmaxMode.LOCOMOTION], env_steps,
                        self.loco_objectives, self.loco_actor_critic, self.train_loco_critic,
                        self.loco_critic_step, self.loco_critic_summaries,
                    )
                else:
                    raise NotImplementedError  # train locomotion with self imitation from trajectories

        with timing.timeit('train_curiosity'):
            self.curiosity.train(buffer, env_steps, agent=self)

        return step

    def _learn_loop(self, multi_env):
        """Main training loop."""
        step, env_steps = self.session.run([self.actor_step, self.total_env_steps])

        observations, goals = self._get_observations(multi_env.reset())
        infos = multi_env.info()

        buffer = TmaxPPOBuffer()

        # separate buffer for complete episode trajectories
        trajectory_buffer = TmaxTrajectoryBuffer(multi_env.num_envs)
        self.curiosity.set_trajectory_buffer(trajectory_buffer)

        tmax_mgr = self.tmax_mgr
        modes = tmax_mgr.initialize(observations, infos, env_steps)

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
                    neighbors, num_neigh = self.tmax_mgr.get_neighbors()

                    with timing.add_time('policy'):
                        actions, action_probs, values, masks, policy_goals = self.policy_step(
                            observations, goals, neighbors, num_neigh,
                        )

                    # wait for all the workers to complete an environment step
                    with timing.add_time('env_step'):
                        env_obs, rewards, dones, infos = multi_env.step(actions)

                    new_obs, new_goals = self._get_observations(env_obs)
                    trajectory_buffer.add(observations, actions, dones, modes=modes)

                    with timing.add_time('tmax'):
                        rewards, dones, modes = tmax_mgr.update(
                            observations, new_obs, rewards, dones, infos, env_steps, timing,
                        )

                    # add experience from all environments to the current buffer(s)
                    buffer.add(
                        observations, policy_goals, actions, action_probs,
                        rewards, dones, values,
                        neighbors, num_neigh, modes, masks,
                    )
                    observations, goals = new_obs, new_goals

                    num_steps += num_env_steps(infos)

                # last step values are required for TD-return calculation
                neighbors, num_neigh = tmax_mgr.get_neighbors()
                _, _, values, _, _ = self.policy_step(observations, goals, neighbors, num_neigh)
                buffer.values.append(values)

            env_steps += num_steps

            # calculate discounted returns and GAE
            buffer.finalize_batch(self.params.gamma, self.params.gae_lambda)

            with timing.timeit('train'):
                step = self._train_tmax(step, buffer, env_steps, timing)

            avg_reward = multi_env.calc_avg_rewards(n=self.params.stats_episodes)
            avg_length = multi_env.calc_avg_episode_lengths(n=self.params.stats_episodes)
            fps = num_steps / (time.time() - batch_start)

            self._maybe_print(step, env_steps, avg_reward, avg_length, fps, timing)
            self._maybe_aux_summaries(env_steps, avg_reward, avg_length, fps)
            self._maybe_tmax_summaries(tmax_mgr, env_steps)
            self._maybe_update_avg_reward(avg_reward, multi_env.stats_num_episodes())
            self._maybe_trajectory_summaries(trajectory_buffer, env_steps)
            self.curiosity.additional_summaries(env_steps, self.summary_writer, self.params.stats_episodes)

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

# TODO: better heuristic to grow the persistent map (e.g. 1 edge at a time)
