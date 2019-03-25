from collections import deque
from functools import partial

import numpy as np
import tensorflow as tf

from algorithms.curiosity.curiosity_module import CuriosityModule
from algorithms.curiosity.reachability_curiosity.observation_encoder import ObservationEncoder
from algorithms.curiosity.reachability_curiosity.reachability import ReachabilityNetwork, ReachabilityBuffer
from algorithms.tf_utils import merge_summaries
from algorithms.topological_maps.localization import Localizer
from algorithms.topological_maps.topological_map import TopologicalMap, map_summaries
from utils.timing import Timing
from utils.utils import log


class ReachabilityCuriosityModule(CuriosityModule):
    class Params:
        def __init__(self):
            self.reachable_threshold = 5  # num. of frames between obs, such that one is reachable from the other
            self.unreachable_threshold = 25  # num. of frames between obs, such that one is unreachable from the other
            self.reachability_target_buffer_size = 100000  # target number of training examples to store
            self.reachability_train_epochs = 10
            self.reachability_batch_size = 128
            self.reachability_bootstrap = 1000000
            self.reachability_train_interval = 500000
            self.reachability_symmetric = True  # useful in 3D environments like Doom and DMLab

            self.new_landmark_threshold = 0.9  # condition for considering current observation a "new landmark"
            self.loop_closure_threshold = 0.6  # condition for graph loop closure (finding new edge)
            self.map_expansion_reward = 0.4  # reward for finding new vertex
            self.reachability_dense_reward = True

    def __init__(self, env, params):
        self.params = params

        self.initialized = False

        self.reachability = ReachabilityNetwork(env, params)

        self._add_summaries()
        self.summaries = merge_summaries(collections=['reachability'])

        self.step = tf.Variable(0, trainable=False, dtype=tf.int64, name='reach_step')
        reach_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='reach_opt')
        self.train_reachability = reach_opt.minimize(self.reachability.loss, global_step=self.step)

        self.trajectory_buffer = None
        self.reachability_buffer = ReachabilityBuffer(self.params)

        self.episodic_maps = None
        self.current_episode_bonus = None
        self.episode_bonuses = deque([])

        self.obs_encoder = ObservationEncoder(encode_func=self.reachability.encode_observation)
        self.localizer = Localizer(self.params, self.obs_encoder)

        self.last_trained = 0

    def _add_summaries(self):
        with tf.name_scope('reachability'):
            reachability_scalar = partial(tf.summary.scalar, collections=['reachability'])
            reachability_scalar('reach_loss', self.reachability.reach_loss)
            reachability_scalar('reach_correct', self.reachability.correct)
            reachability_scalar('reg_loss', self.reachability.reg_loss)

    # noinspection PyProtectedMember
    def _train_reachability(self, data, env_steps, agent):
        timing = Timing()

        with timing.timeit('get_buffer'):
            buffer = data.get_buffer()
        assert len(buffer) <= self.params.reachability_target_buffer_size

        batch_size = self.params.reachability_batch_size
        summary = None
        reach_step = self.step.eval(session=agent.session)

        prev_loss = 1e10
        num_epochs = self.params.reachability_train_epochs

        log.info('Training reachability %d pairs, batch %d, epochs %d, %s', len(buffer), batch_size, num_epochs, timing)

        for epoch in range(num_epochs):
            losses = []
            buffer.shuffle_data()
            obs_first, obs_second, labels = buffer.obs_first, buffer.obs_second, buffer.labels

            for i in range(0, len(obs_first) - 1, batch_size):
                with_summaries = agent._should_write_summaries(reach_step) and summary is None
                summaries = [self.summaries] if with_summaries else []

                start, end = i, i + batch_size

                result = agent.session.run(
                    [self.reachability.loss, self.train_reachability] + summaries,
                    feed_dict={
                        self.reachability.ph_obs_first: obs_first[start:end],
                        self.reachability.ph_obs_second: obs_second[start:end],
                        self.reachability.ph_labels: labels[start:end],
                    }
                )

                reach_step += 1
                agent._maybe_save(reach_step, env_steps)
                losses.append(result[0])

                if with_summaries:
                    summary = result[-1]
                    agent.summary_writer.add_summary(summary, global_step=env_steps)

            # check loss improvement at the end of each epoch, early stop if necessary
            avg_loss = np.mean(losses)
            if avg_loss >= prev_loss:
                log.info('Early stopping after %d epochs because reachability did not improve', epoch + 1)
                log.info('Was %.4f now %.4f, ratio %.3f', prev_loss, avg_loss, avg_loss / prev_loss)
                break
            prev_loss = avg_loss

        self.last_trained = env_steps

    def generate_bonus_rewards(self, session, obs, next_obs, actions, dones, infos):
        if self.episodic_maps is None:
            self.current_episode_bonus = np.zeros(self.params.num_envs)  # for statistics
            self.episodic_maps = []
            for i in range(self.params.num_envs):
                self.episodic_maps.append(TopologicalMap(obs[i], directed_graph=False, initial_info=infos[i]))

        for i in range(self.params.num_envs):
            if dones[i]:
                self.episodic_maps[i].reset(next_obs[i], infos[i])
                self.episode_bonuses.append(self.current_episode_bonus[i])
                self.current_episode_bonus[i] = 0

        bonuses = np.zeros(self.params.num_envs)

        if self.initialized:
            def on_new_landmark(env_i):
                bonuses[env_i] += self.params.map_expansion_reward

            distances_to_memory = self.localizer.localize(
                session, next_obs, infos, self.episodic_maps, self.reachability, on_new_landmark=on_new_landmark,
            )
            assert len(distances_to_memory) == len(next_obs)
            threshold = self.params.new_landmark_threshold
            dense_rewards = np.array([
                0.0 if done else dist - threshold for (dist, done) in zip(distances_to_memory, dones)
            ])
            dense_rewards *= 0.1  # scaling factor

            if self.params.reachability_dense_reward:
                bonuses += dense_rewards

        self.current_episode_bonus += bonuses
        return bonuses

    def train(self, buffer, env_steps, agent):
        self.reachability_buffer.extract_data(self.trajectory_buffer.complete_trajectories)

        if env_steps - self.last_trained > self.params.reachability_train_interval:
            if self.reachability_buffer.has_enough_data():
                self._train_reachability(self.reachability_buffer, env_steps, agent)

                # discard old experience
                self.reachability_buffer.reset()

                # invalidate observation features because reachability network has changed
                self.obs_encoder.reset()

        if env_steps > self.params.reachability_bootstrap:
            self.initialized = True

    def set_trajectory_buffer(self, trajectory_buffer):
        self.trajectory_buffer = trajectory_buffer

    def is_initialized(self):
        return self.initialized

    def additional_summaries(self, env_steps, summary_writer, stats_episodes):
        maps = self.episodic_maps
        if not self.initialized or maps is None:
            return

        summary = tf.Summary()
        section = 'curiosity_reachability'

        def curiosity_summary(tag, value):
            summary.value.add(tag=f'{section}/{tag}', simple_value=float(value))

        # log bonuses per episode
        if len(self.episode_bonuses) > 0:
            while len(self.episode_bonuses) > stats_episodes:
                self.episode_bonuses.popleft()
            avg_episode_bonus = sum(self.episode_bonuses) / len(self.episode_bonuses)
            curiosity_summary('avg_episode_bonus', avg_episode_bonus)

        summary_writer.add_summary(summary, env_steps)

        map_summaries(maps, env_steps, summary_writer, section)
        summary_writer.flush()
