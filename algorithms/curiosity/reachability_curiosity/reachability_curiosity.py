import numpy as np
from functools import partial

import tensorflow as tf

from algorithms.curiosity.curiosity_module import CuriosityModule
from algorithms.curiosity.reachability_curiosity.observation_encoder import ObservationEncoder
from algorithms.curiosity.reachability_curiosity.reachability import ReachabilityNetwork, ReachabilityBuffer
from algorithms.tf_utils import merge_summaries
from algorithms.topological_maps.localization import Localizer
from algorithms.topological_maps.topological_map import TopologicalMap
from utils.utils import log


class ReachabilityCuriosityModule(CuriosityModule):
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
        buffer = data.buffer

        batch_size = self.params.reachability_batch_size
        summary = None
        reach_step = self.step.eval(session=agent.session)

        prev_loss = 1e10
        num_epochs = self.params.reachability_train_epochs

        log.info('Training reachability %d pairs, batch %d, epochs %d', len(buffer), batch_size, num_epochs)

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

        # invalidate observation features because reachability network has changed
        self.obs_encoder.reset()

        self.last_trained = env_steps

    def generate_bonus_rewards(self, session, obs, next_obs, actions, dones, infos):
        if self.episodic_maps is None:
            self.episodic_maps = []
            for i in range(self.params.num_envs):
                self.episodic_maps.append(TopologicalMap(obs[i], True, infos[i]))

        for i in range(self.params.num_envs):
            if dones[i]:
                self.episodic_maps[i].reset(obs[i], infos[i])

        if self.initialized:
            bonuses = self.localizer.localize(session, obs, infos, self.episodic_maps, self.reachability)
        else:
            bonuses = np.zeros(self.params.num_envs)

        return bonuses

    def train(self, buffer, env_steps, agent):
        self.reachability_buffer.extract_data(self.trajectory_buffer.complete_trajectories)

        if self.reachability_buffer.has_enough_data():
            if env_steps - self.last_trained > self.params.reachability_train_interval:
                self._train_reachability(self.reachability_buffer, env_steps, agent)

        if env_steps > self.params.reachability_bootstrap:
            self.initialized = True

    def set_trajectory_buffer(self, trajectory_buffer):
        self.trajectory_buffer = trajectory_buffer

    def is_initialized(self):
        return self.initialized
