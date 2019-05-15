from collections import deque
from functools import partial

import numpy as np
import tensorflow as tf

from algorithms.curiosity.curiosity_module import CuriosityModule
from algorithms.curiosity.ecr.episodic_memory import EpisodicMemory
from algorithms.distance.distance import DistanceNetwork, DistanceBuffer
from utils.utils import log


class ECRModule(CuriosityModule):
    class Params:
        def __init__(self):
            self.reachable_threshold = 5  # num. of frames between obs, such that one is reachable from the other
            self.unreachable_threshold = 25  # num. of frames between obs, such that one is unreachable from the other
            self.distance_target_buffer_size = 200000  # target number of training examples to store
            self.distance_train_epochs = 8
            self.distance_batch_size = 128
            self.distance_bootstrap = 2000000
            self.distance_train_interval = 1000000
            self.distance_symmetric = True  # useful in 3D environments like Doom and DMLab

            self.episodic_memory_size = 200
            self.ecr_dense_reward = True
            self.dense_reward_scaling_factor = 0.1
            self.dense_reward_threshold = 0.5
            self.add_to_memory_threshold = 0.9

            self.memory_expansion_reward = 0.2  # reward for finding new memory entry

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
        obs_enc = self.distance.encode_observation(session, obs)
        if self.episodic_memories is None:
            self.current_episode_bonus = np.zeros(self.params.num_envs)  # for statistics
            self.episodic_memories = []
            for i in range(self.params.num_envs):
                self.episodic_memories.append(EpisodicMemory(self.params, obs_enc[i]))

        next_obs_enc = self.distance.encode_observation(session, next_obs)
        assert len(next_obs_enc) == len(self.episodic_memories)

        for env_i in range(self.params.num_envs):
            if dones[env_i]:
                self.episodic_memories[env_i].reset(next_obs_enc[env_i])
                self.episode_bonuses.append(self.current_episode_bonus[env_i])
                self.current_episode_bonus[env_i] = 0

        bonuses = np.zeros(self.params.num_envs)

        if self.initialized:
            memory_extended = []
            next_obs_enc_extended = []
            memory_lengths = []
            for env_i, memory in enumerate(self.episodic_memories):
                if not dones[env_i]:
                    memlen = len(memory.arr)
                    memory_extended.extend(memory.arr)
                    next_obs_enc_extended.extend([next_obs_enc[env_i]] * memlen)
                    memory_lengths.append(memlen)
                else:
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
                if not dones[env_i] and memlen > 0:
                    new_count = count + memlen
                    distances_to_memory.append(np.percentile(batch_distances[count:new_count], 10))
                    count = new_count
                else:
                    distances_to_memory.append(0)

            assert len(distances_to_memory) == len(next_obs_enc)

            sparse_rewards = np.zeros(self.params.num_envs)
            for i, dist in enumerate(distances_to_memory):
                if dist > self.params.add_to_memory_threshold:
                    self.episodic_memories[i].add(next_obs_enc[i])
                    sparse_rewards[i] += self.params.memory_expansion_reward

            bonuses += sparse_rewards

            dense_rewards = np.array([
                0.0 if done else dist - self.params.dense_reward_threshold for (dist, done) in zip(distances_to_memory, dones)
            ])

            dense_rewards *= self.params.dense_reward_scaling_factor

            if self.params.ecr_dense_reward:
                assert len(dense_rewards) == len(bonuses)
                bonuses += dense_rewards


        self.current_episode_bonus += bonuses
        return bonuses

    def train(self, buffer, env_steps, agent):
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
        summary_writer.flush()


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
