import copy
import time
from collections import deque

import numpy as np
import tensorflow as tf

from algorithms.curiosity.curiosity_module import CuriosityModule
from algorithms.distance.distance import DistanceNetwork, DistanceNetworkParams, DistanceBuffer
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
            self.ecr_dense_reward = False

            self.expand_explored_region = False
            self.expand_explored_region_frames = 4000000

    def __init__(self, env, params):
        self.params = params

        self.initialized = False

        self.distance = DistanceNetwork(env, params)

        self.trajectory_buffer = None
        self.distance_buffer = DistanceBuffer(self.params)

        self.episodic_maps = None
        self.current_episode_bonus = None
        self.episode_bonuses = deque([])

        self.localizer = Localizer(self.params)

        self.past_maps = deque([], maxlen=200)
        self.last_explored_region_update = self.params.distance_bootstrap
        self.explored_region_map = None

        self._last_trained = 0
        self._last_map_summary = 0

    def generate_bonus_rewards(self, session, obs, next_obs, actions, dones, infos):
        if self.episodic_maps is None:
            self.current_episode_bonus = np.zeros(self.params.num_envs)  # for statistics
            self.episodic_maps = []
            for i in range(self.params.num_envs):
                self.episodic_maps.append(TopologicalMap(obs[i], directed_graph=False, initial_info=infos[i]))

        for i in range(self.params.num_envs):
            if dones[i]:
                if self.params.expand_explored_region:
                    # save last n maps for later use
                    self.past_maps.append(copy.deepcopy(self.episodic_maps[i]))

                self.episode_bonuses.append(self.current_episode_bonus[i])
                self.current_episode_bonus[i] = 0

                if self.explored_region_map is not None:
                    assert self.params.expand_explored_region

                    # set the episodic map to be the map of the explored region, so we don't receive any more reward
                    # for seeing what we've already explored
                    self.episodic_maps[i] = copy.deepcopy(self.explored_region_map)
                else:
                    # we don't have a map of explored region, so reset episodic memory to zero
                    self.episodic_maps[i].reset(next_obs[i], infos[i])

                self.episodic_maps[i].new_episode()

        bonuses = np.zeros(self.params.num_envs)

        if self.initialized:
            # noinspection PyUnusedLocal
            def on_new_landmark(env_i, new_landmark_idx):
                bonuses[env_i] += self.params.map_expansion_reward

            distances_to_memory = self.localizer.localize(
                session, next_obs, infos, self.episodic_maps, self.distance, on_new_landmark=on_new_landmark,
            )
            assert len(distances_to_memory) == len(next_obs)
            threshold = 0.5
            dense_rewards = np.array([
                0.0 if done else dist - threshold for (dist, done) in zip(distances_to_memory, dones)
            ])
            dense_rewards *= 0.1  # scaling factor

            if self.params.ecr_dense_reward:
                bonuses += dense_rewards

        self.current_episode_bonus += bonuses
        return bonuses

    def train(self, buffer, env_steps, agent):
        self.distance_buffer.extract_data(self.trajectory_buffer.complete_trajectories)

        if env_steps - self._last_trained > self.params.distance_train_interval:
            if self.distance_buffer.has_enough_data():
                self.distance.train(self.distance_buffer.get_buffer(), env_steps, agent)
                self._last_trained = env_steps

                # discard old experience
                self.distance_buffer.reset()

                # invalidate observation features because distance network has changed
                self.distance.obs_encoder.reset()

        if env_steps > self.params.distance_bootstrap and not self.is_initialized():
            log.debug('Curiosity is initialized @ %d steps!', env_steps)
            self.initialized = True

        if self.params.expand_explored_region:
            if env_steps - self.last_explored_region_update >= self.params.expand_explored_region_frames:
                if len(self.past_maps) >= self.past_maps.maxlen:
                    map_sizes = []
                    for i, m in enumerate(self.past_maps):
                        map_sizes.append((m.num_landmarks(), i))

                    map_sizes.sort()
                    median_map_idx = map_sizes[len(map_sizes) // 2][1]
                    median_map = self.past_maps[median_map_idx]

                    log.debug(
                        'Select map %d with %d landmarks as our new map of explored region @ %d frames',
                        median_map_idx, median_map.num_landmarks(), env_steps,
                    )
                    self.explored_region_map = copy.deepcopy(median_map)
                    self.last_explored_region_update = env_steps

                    checkpoint_dir = model_dir(agent.params.experiment_dir())
                    self.explored_region_map.save_checkpoint(
                        checkpoint_dir, map_img=agent.map_img, coord_limits=agent.coord_limits, verbose=True,
                    )

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

        summary_writer.add_summary(summary, env_steps)

        time_since_last = time.time() - self._last_map_summary
        map_summary_rate_seconds = 60
        if time_since_last > map_summary_rate_seconds:
            map_img = kwargs.get('map_img')
            coord_limits = kwargs.get('coord_limits')
            map_summaries(maps, env_steps, summary_writer, section, map_img, coord_limits)
            summary_writer.flush()

            if self.explored_region_map is not None:
                map_summaries(
                    [self.explored_region_map], env_steps, summary_writer, 'explored_region', map_img, coord_limits,
                )

            self._last_map_summary = time.time()
