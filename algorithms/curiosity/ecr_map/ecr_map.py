import copy
import math
import time
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from algorithms.curiosity.curiosity_module import CuriosityModule
from algorithms.distance.distance import DistanceNetwork, DistanceNetworkParams, DistanceBuffer
from algorithms.distance.distance_oracle import DistanceOracle
from algorithms.topological_maps.localization import Localizer
from algorithms.topological_maps.topological_map import TopologicalMap, map_summaries
from utils.utils import log, model_dir


class ECRMapModule(CuriosityModule):
    class Params(DistanceNetworkParams):
        def __init__(self):
            DistanceNetworkParams.__init__(self)

            self.new_landmark_threshold = 0.9  # condition for considering current observation a "new landmark"
            self.loop_closure_threshold = 0.6  # condition for graph loop closure (finding new edge)
            self.map_expansion_reward = 0.2  # reward for finding new vertex
            self.per_step_intrinsic_reward = -0.02  # to make cumulative reward negative (to be attracted to goals)

            self.revisiting_penalty = 0.0
            self.revisiting_threshold = 0.2
            self.revisit_num_frames = 5

            self.ecr_map_dense_reward = False
            self.ecr_map_sparse_reward = True

            self.ecr_map_adaptive_reward = True

            self.expand_explored_region = False
            self.expand_explored_region_frames = 4000000

            self.oracle_distance = False  # debug mode, using ground truth coordinates from the env.

    def __init__(self, env, params):
        self.params = params

        self.initialized = False

        if self.params.oracle_distance:
            self.distance = DistanceOracle(env, params)
        else:
            self.distance = DistanceNetwork(env, params)

        self.trajectory_buffer = None
        self.distance_buffer = DistanceBuffer(self.params)

        self.episodic_maps = [None] * params.num_envs
        self.current_episode_bonus = np.zeros(self.params.num_envs)
        self.episode_bonuses = deque([])

        self.localizer = Localizer(self.params)

        self.past_maps = deque([], maxlen=200)
        self.last_explored_region_update = self.params.distance_bootstrap
        self.explored_region_map = None

        self.episode_frames = [0] * self.params.num_envs

        # statistics for adaptive reward threshold
        self.frames_analyzed = 0
        self.landmarks_generated = 0
        self.new_landmark_threshold = self.params.new_landmark_threshold
        self.loop_closure_threshold = self.params.loop_closure_threshold

        self._last_trained = 0
        self._last_map_summary = 0

    def initialize(self, session):
        # restore only distance network if we have checkpoint for it
        if self.params.distance_network_checkpoint is not None:
            log.debug('Restoring distance net variables from %s', self.params.distance_network_checkpoint)
            variables = slim.get_variables_to_restore()
            distance_net_variables = [v for v in variables if v.name.split('/')[0] == 'distance']
            distance_net_saver = tf.train.Saver(distance_net_variables)
            distance_net_saver.restore(
                session, tf.train.latest_checkpoint(self.params.distance_network_checkpoint),
            )
            self.initialized = True
            log.debug('Done loading distance network from checkpoint!')

    def generate_bonus_rewards(self, session, obs, next_obs, actions, dones, infos, mask=None):
        if self.explored_region_map is None:
            self.explored_region_map = TopologicalMap(obs[0], directed_graph=False, initial_info=infos[0])

        for i, episodic_map in enumerate(self.episodic_maps):
            if episodic_map is None:
                # noinspection PyTypeChecker
                self.episodic_maps[i] = copy.deepcopy(self.explored_region_map)

        for i in range(self.params.num_envs):
            if dones[i]:
                if self.params.expand_explored_region:
                    # save last n maps for later use
                    self.past_maps.append(copy.deepcopy(self.episodic_maps[i]))

                self.episode_bonuses.append(self.current_episode_bonus[i])
                self.current_episode_bonus[i] = 0

                if self.explored_region_map is not None:
                    # set the episodic map to be the map of the explored region, so we don't receive any more reward
                    # for seeing what we've already explored
                    # noinspection PyTypeChecker
                    self.episodic_maps[i] = copy.deepcopy(self.explored_region_map)
                    for node in self.episodic_maps[i].graph.nodes:
                        self.episodic_maps[i].graph.nodes[node]['added_at'] = -1
                else:
                    # we don't have a map of explored region, so reset episodic memory to zero
                    self.episodic_maps[i].reset(next_obs[i], infos[i])

                self.episodic_maps[i].new_episode()

                self.episode_frames[i] = 0
            else:
                self.episode_frames[i] += 1
                self.frames_analyzed += 1

        frames = self.episode_frames
        bonuses = np.full(self.params.num_envs, fill_value=self.params.per_step_intrinsic_reward)
        with_sparse_reward = self.params.ecr_map_sparse_reward

        if self.initialized:
            # noinspection PyUnusedLocal
            def on_new_landmark(env_i_, new_landmark_idx):
                if with_sparse_reward:
                    bonuses[env_i_] += self.params.map_expansion_reward
                    self.landmarks_generated += 1

            if mask is None:
                maps = self.episodic_maps
            else:
                maps = [self.episodic_maps[i] if mask[i] else None for i in range(len(mask))]

            self.localizer.new_landmark_threshold = self.new_landmark_threshold
            self.localizer.loop_closure_threshold = self.loop_closure_threshold
            distances_to_memory = self.localizer.localize(
                session, next_obs, infos, maps, self.distance, frames=frames, on_new_landmark=on_new_landmark,
            )

            if frames is not None:
                for env_i, m in enumerate(maps):
                    if m is None:
                        continue

                    if distances_to_memory[env_i] < self.params.revisiting_threshold:
                        added_at = m.graph.nodes[m.curr_landmark_idx].get('added_at', -1)
                        if added_at == -1:
                            continue

                        if frames[env_i] - added_at > self.params.revisit_num_frames:
                            bonuses[env_i] += self.params.revisiting_penalty

            # if bonuses[0] > 0:
            #     log.warning('Distances to memory: %.3f, bonuses: %.3f', distances_to_memory[0], bonuses[0])
            # else:
            #     log.info('Distances to memory: %.3f, bonuses: %.3f', distances_to_memory[0], bonuses[0])

            assert len(distances_to_memory) == len(next_obs)
            threshold = 1.0
            dense_rewards = np.array([
                0.0 if done else dist - threshold for (dist, done) in zip(distances_to_memory, dones)
            ])
            dense_rewards *= 0.05  # scaling factor

            if self.params.ecr_map_dense_reward:
                for i in range(self.params.num_envs):
                    if maps[i] is not None:
                        bonuses[i] += dense_rewards[i]

            if math.nan in bonuses:
                log.error('Bonuses: %r', bonuses)
                log.error('NaN values in bonus array!')

        self.current_episode_bonus += bonuses

        if self.params.ecr_map_adaptive_reward:
            if self.frames_analyzed >= 50000:
                ratio = self.landmarks_generated / self.frames_analyzed
                if ratio < 25 / 1000:
                    # make landmarks easier to find
                    self.new_landmark_threshold *= 0.95
                    self.loop_closure_threshold = 0.5 * self.new_landmark_threshold
                    log.info(
                        'Decreased landmark threshold to %.3f (%.3f)',
                        self.new_landmark_threshold, self.loop_closure_threshold,
                    )
                elif ratio > 40 / 1000:
                    not_far_probability = 1.0 - self.new_landmark_threshold
                    not_far_probability *= 0.9  # decrease minimum probability that new landmark is not "far"
                    self.new_landmark_threshold = 1.0 - not_far_probability
                    self.loop_closure_threshold = 0.5 * self.new_landmark_threshold
                    log.info(
                        'Increased landmark threshold to %.3f (%.3f)',
                        self.new_landmark_threshold, self.loop_closure_threshold,
                    )
                else:
                    log.info('Landmark threshold unchanged, ratio %.3f', ratio)

                self.frames_analyzed = 0
                self.landmarks_generated = 0

        return bonuses

    def _expand_explored_region(self, env_steps, agent):
        if not self.params.expand_explored_region:
            return

        if env_steps - self.last_explored_region_update < self.params.expand_explored_region_frames:
            return

        if len(self.past_maps) <= 0:
            return

        max_landmarks_idx = 0
        for i, m in enumerate(self.past_maps):
            if m.num_landmarks() > self.past_maps[max_landmarks_idx].num_landmarks():
                max_landmarks_idx = i

        biggest_map = self.past_maps[max_landmarks_idx]
        log.debug('Biggest map %d with %d landmarks', max_landmarks_idx, biggest_map.num_landmarks())

        existing_nodes = set(self.explored_region_map.graph.nodes)
        node_distances = biggest_map.topological_distances(0)

        node_distances = [(dist, idx) for idx, dist in node_distances.items()]
        node_distances.sort()

        num_to_add, num_added = 5, 0
        new_map_nodes = []
        for dist, idx in node_distances:
            if idx in existing_nodes:
                new_map_nodes.append(idx)
                log.debug('Keep node %d, it is already in the map of explored region', idx)
                continue

            new_map_nodes.append(idx)
            log.debug('Adding new node %d with topological distance %d', idx, dist)
            num_added += 1
            if num_added >= num_to_add:
                break

        log.debug('List of nodes in the new map %r', new_map_nodes)

        self.explored_region_map.graph = biggest_map.graph.subgraph(new_map_nodes).copy()
        self.explored_region_map.new_episode()

        self.last_explored_region_update = env_steps

        checkpoint_dir = model_dir(agent.params.experiment_dir())
        self.explored_region_map.save_checkpoint(
            checkpoint_dir, map_img=agent.map_img, coord_limits=agent.coord_limits, verbose=True,
        )

        self.past_maps.clear()

    def train(self, latest_batch_of_experience, env_steps, agent):
        # latest batch of experience is not used here

        if self.params.distance_network_checkpoint is None:
            # don't train distance net if it's already provided

            self.distance_buffer.extract_data(self.trajectory_buffer.complete_trajectories)

            if env_steps - self._last_trained > self.params.distance_train_interval:
                if self.distance_buffer.has_enough_data():
                    self.distance.train(self.distance_buffer.buffer, env_steps, agent)
                    self._last_trained = env_steps

                    # discard old experience
                    self.distance_buffer.reset()

                    # invalidate observation features because distance network has changed
                    self.distance.obs_encoder.reset()

        if env_steps > self.params.distance_bootstrap and not self.is_initialized():
            log.debug('Curiosity is initialized @ %d steps!', env_steps)
            self.initialized = True

        self._expand_explored_region(env_steps, agent)

    def set_trajectory_buffer(self, trajectory_buffer):
        self.trajectory_buffer = trajectory_buffer

    def is_initialized(self):
        return self.initialized

    def additional_summaries(self, env_steps, summary_writer, stats_episodes, **kwargs):
        maps = self.episodic_maps
        if not self.initialized or maps is None:
            return

        summary = tf.Summary()
        section = 'ecr_map'

        def curiosity_summary(tag, value):
            summary.value.add(tag=f'{section}/{tag}', simple_value=float(value))

        # log bonuses per episode
        if len(self.episode_bonuses) > 0:
            while len(self.episode_bonuses) > stats_episodes:
                self.episode_bonuses.popleft()
            avg_episode_bonus = sum(self.episode_bonuses) / len(self.episode_bonuses)
            curiosity_summary('avg_episode_bonus', avg_episode_bonus)

        if self.params.expand_explored_region:
            explored_region_size = 0 if self.explored_region_map is None else self.explored_region_map.num_landmarks()
            curiosity_summary('explored_region_size', explored_region_size)

        curiosity_summary('new_landmark_threshold', self.new_landmark_threshold)
        curiosity_summary('loop_closure_threshold', self.loop_closure_threshold)

        summary_writer.add_summary(summary, env_steps)

        time_since_last = time.time() - self._last_map_summary
        map_summary_rate_seconds = 120
        if time_since_last > map_summary_rate_seconds:
            map_img = kwargs.get('map_img')
            coord_limits = kwargs.get('coord_limits')
            map_summaries(maps, env_steps, summary_writer, section, map_img, coord_limits, is_sparse=True)

            if self.explored_region_map is not None and self.params.expand_explored_region:
                map_summaries(
                    [self.explored_region_map], env_steps, summary_writer, 'explored_region', map_img, coord_limits,
                    is_sparse=True,
                )

            summary_writer.flush()
            self._last_map_summary = time.time()
