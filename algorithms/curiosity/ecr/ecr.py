import time
from collections import deque

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim

from algorithms.curiosity.curiosity_module import CuriosityModule
from algorithms.curiosity.ecr.episodic_memory import EpisodicMemory
from algorithms.distance.distance import DistanceNetwork, DistanceBuffer, DistanceNetworkParams
from algorithms.topological_maps.topological_map import map_summaries, TopologicalMap, get_position, get_angle
from utils.timing import Timing
from utils.utils import log


class ECRModule(CuriosityModule):
    class Params(DistanceNetworkParams):
        def __init__(self):
            DistanceNetworkParams.__init__(self)

            self.episodic_memory_size = 600
            self.ecr_memory_sample_size = 200

            self.novelty_threshold = 0.0  # condition for adding current observation to memory

            self.ecr_sparse_reward = False
            self.sparse_reward_size = 0.2  # optional reward for finding new memory entry

            self.ecr_dense_reward = True
            self.dense_reward_scaling_factor = 1.0
            self.dense_reward_threshold = 0.5

            self.ecr_reset_memory = True

    def __init__(self, env, params):
        self.params = params

        self.initialized = False

        self.distance = DistanceNetwork(env, params)

        self.trajectory_buffer = None
        self.distance_buffer = DistanceBuffer(self.params)

        self.episodic_memories = None
        self.current_episode_bonus = None
        self.episode_bonuses = deque([])

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

    def generate_bonus_rewards(self, session, obs, next_obs, actions, dones, infos):
        if self.episodic_memories is None:
            obs_enc = self.distance.encode_observation(session, obs)
            self.current_episode_bonus = np.zeros(self.params.num_envs)  # for statistics
            self.episodic_memories = []
            for i in range(self.params.num_envs):
                self.episodic_memories.append(EpisodicMemory(self.params, obs_enc[i], infos[i]))

        next_obs_enc = self.distance.encode_observation(session, next_obs)
        assert len(next_obs_enc) == len(self.episodic_memories)

        for env_i in range(self.params.num_envs):
            if dones[env_i]:
                if self.params.ecr_reset_memory:
                    self.episodic_memories[env_i].reset(next_obs_enc[env_i], infos[env_i])
                self.episode_bonuses.append(self.current_episode_bonus[env_i])
                self.current_episode_bonus[env_i] = 0

        bonuses = np.zeros(self.params.num_envs)

        if self.initialized:
            memory_extended = []
            next_obs_enc_extended = []
            memory_lengths = []
            sample_landmarks = []
            for env_i, memory in enumerate(self.episodic_memories):
                if not dones[env_i]:
                    embeddings, landmark_indices = memory.sample_landmarks(self.params.ecr_memory_sample_size)
                    sample_landmarks.append(landmark_indices)
                    memory_extended.extend(embeddings)
                    next_obs_enc_extended.extend([next_obs_enc[env_i]] * len(embeddings))
                    memory_lengths.append(len(embeddings))
                else:
                    sample_landmarks.append([])
                    memory_lengths.append(0)

            assert len(memory_extended) == len(next_obs_enc_extended)

            batch_distances = []
            batch_size = 1024
            for i in range(0, len(next_obs_enc_extended), batch_size):
                start, end = i, i + batch_size

                distances_batch = self.distance.distances(
                    session,
                    obs_first_encoded=memory_extended[start:end],
                    obs_second_encoded=next_obs_enc_extended[start:end]
                )
                batch_distances.extend(distances_batch)

            count = 0
            distances_to_memory = []
            for env_i, memlen in enumerate(memory_lengths):
                new_count = count + memlen

                if not dones[env_i] and memlen > 0:
                    this_env_distances = batch_distances[count:new_count]
                    distance_percentile = self.episodic_memories[env_i].distance_percentile(
                        this_env_distances, sample_landmarks[env_i], percentile=10,
                    )
                    distances_to_memory.append(distance_percentile)
                else:
                    distances_to_memory.append(0)

                count = new_count

            assert len(distances_to_memory) == len(next_obs_enc)

            dense_rewards = np.array([
                0.0 if done else dist - self.params.dense_reward_threshold
                for (dist, done) in zip(distances_to_memory, dones)
            ])

            dense_rewards *= self.params.dense_reward_scaling_factor

            if self.params.ecr_dense_reward:
                assert len(dense_rewards) == len(bonuses)
                bonuses += dense_rewards

            sparse_rewards = np.zeros(self.params.num_envs)
            for i, rew in enumerate(dense_rewards):
                if rew > self.params.novelty_threshold:
                    self.episodic_memories[i].add(next_obs_enc[i], infos[i])
                    sparse_rewards[i] += self.params.sparse_reward_size

            if self.params.ecr_sparse_reward:
                assert len(sparse_rewards) == len(bonuses)
                bonuses += sparse_rewards

        self.current_episode_bonus += bonuses
        return bonuses

    def train(self, buffer, env_steps, agent):
        if self.params.distance_network_checkpoint is None:

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

    def set_trajectory_buffer(self, trajectory_buffer):
        self.trajectory_buffer = trajectory_buffer

    def is_initialized(self):
        return self.initialized

    def additional_summaries(self, env_steps, summary_writer, stats_episodes, **kwargs):
        memories = self.episodic_memories
        if not self.initialized or memories is None:
            return

        summary = tf.Summary()
        section = 'ecr'

        def curiosity_summary(tag, value):
            summary.value.add(tag=f'{section}/{tag}', simple_value=float(value))

        # log bonuses per episode
        if len(self.episode_bonuses) > 0:
            while len(self.episode_bonuses) > stats_episodes:
                self.episode_bonuses.popleft()
            avg_episode_bonus = sum(self.episode_bonuses) / len(self.episode_bonuses)
            curiosity_summary('avg_episode_bonus', avg_episode_bonus)

        summary_writer.add_summary(summary, env_steps)
        buffer_summaries(memories, env_steps, summary_writer, section)
        self.episodic_memory_summary(env_steps, summary_writer, **kwargs)

        summary_writer.flush()

    def episodic_memory_summary(self, env_steps, summary_writer, **kwargs):
        t = Timing()

        with t.timeit('ecr_memory'):
            time_since_last = time.time() - self._last_map_summary
            map_summary_rate_seconds = 120
            if time_since_last <= map_summary_rate_seconds:
                return
            if self.episodic_memories is None:
                return

            env_to_plot = 0
            for env_i, memory in enumerate(self.episodic_memories):
                if len(memory) > len(self.episodic_memories[env_to_plot]):
                    env_to_plot = env_i

            log.info('Visualizing episodic memory for env %d', env_to_plot)
            memory_to_plot = self.episodic_memories[env_to_plot]

            if len(memory_to_plot) <= 0:
                return

            landmark_indices = sorted(memory_to_plot.landmarks.keys())

            m = TopologicalMap(
                memory_to_plot.landmarks[landmark_indices[0]].embedding,
                directed_graph=False,
                initial_info=memory_to_plot.landmarks[landmark_indices[0]].info,
            )

            for lm_idx in landmark_indices[1:]:
                info = memory_to_plot.landmarks[lm_idx].info
                # noinspection PyProtectedMember
                m._add_new_node(
                    obs=memory_to_plot.landmarks[lm_idx].embedding,
                    pos=get_position(info),
                    angle=get_angle(info),
                )

            map_img = kwargs.get('map_img')
            coord_limits = kwargs.get('coord_limits')
            map_summaries([m], env_steps, summary_writer, 'ecr', map_img, coord_limits, is_sparse=True)

            self._last_map_summary = time.time()
        log.info('Took %s', t)


def buffer_summaries(buffers, env_steps, summary_writer, section):
    if None in buffers:
        return
    # summaries related to episodic memory
    num_obs = [len(b) for b in buffers]

    summary = tf.Summary()

    def curiosity_summary(tag, value):
        summary.value.add(tag=f'{section}/{tag}', simple_value=float(value))
    curiosity_summary('avg_num_obs', sum(num_obs) / len(num_obs))
    curiosity_summary('max_num_obs', max(num_obs))
    curiosity_summary('min_num_obs', min(num_obs))

    summary_writer.add_summary(summary, env_steps)
