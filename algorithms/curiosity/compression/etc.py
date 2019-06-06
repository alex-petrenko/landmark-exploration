"""

Exploration through compression.

"""
from collections import deque
from functools import partial

import numpy as np
import tensorflow as tf

from algorithms.curiosity.curiosity_module import CuriosityModule
from algorithms.utils.decoders import make_decoder
from algorithms.utils.encoders import make_encoder, EncoderParams
from algorithms.utils.env_wrappers import main_observation_space
from algorithms.utils.tf_utils import merge_summaries, dense, summary_avg_min_max, image_summaries_rgb


class VAE:
    def __init__(self, ph_obs, obs_space, params, env_steps):
        self.num_latent = params.num_latent
        self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-5)

        enc_params = EncoderParams()
        enc_params.enc_name = 'convnet_84px_bigger'
        enc_params.summary_collections = ['vae']
        encoder = tf.make_template(
            'vae_enc', make_encoder, create_scope_now_=True,
            obs_space=obs_space, regularizer=self.regularizer, enc_params=enc_params,
        )

        obs_enc = encoder(ph_obs)

        obs_normalized = obs_enc.normalized_obs

        # variational bottleneck
        self.z_mu = dense(obs_enc.encoded_input, self.num_latent, activation=None)

        if params.variational:
            self.z_log_sigma_sq = dense(obs_enc.encoded_input, self.num_latent, activation=None)
            sigma = tf.sqrt(tf.exp(self.z_log_sigma_sq))
            self.eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq), mean=0, stddev=1, dtype=tf.float32)
            self.z = self.z_mu + sigma * self.eps
        else:
            self.z = self.z_mu  # pure autoencoder

        decoder = tf.make_template('vae_dec', make_decoder, create_scope_now_=True, name='dec')

        obs_dec = decoder(self.z).decoded

        # reconstruction loss
        # mean across channels
        img_losses = tf.reduce_mean(tf.squared_difference(obs_normalized, obs_dec), axis=3)
        # sum over the entire image
        img_losses = tf.reduce_sum(img_losses, axis=[1, 2])
        # mean across batch
        img_loss = tf.reduce_mean(img_losses)

        # regularization loss
        regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # total loss
        reconst_coeff = 1.0
        self.losses = reconst_coeff * img_losses

        if params.variational:
            # latent loss
            # KL divergence: measure the difference between two distributions
            # Here we measure the divergence between the latent distribution and N(0, 1)
            latent_losses = -0.5 * tf.reduce_sum(
                1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1,
            )

            start_kl_coeff = 0.0
            final_kl_coeff = 1.0
            kl_decay_env_steps = 5e8

            # decay KL component to prevent posterior collapse
            kl_coeff = tf.train.polynomial_decay(
                start_kl_coeff,
                env_steps,
                kl_decay_env_steps,
                final_kl_coeff,
                power=1.0,
                cycle=False,
            )
            self.losses += kl_coeff * latent_losses
            latent_loss = tf.reduce_mean(latent_losses)
        else:
            kl_coeff = latent_loss = 0.0

        self.loss = tf.reduce_mean(self.losses) + regularization_loss

        vae_scalar = partial(tf.summary.scalar, collections=['vae'])

        with tf.name_scope('vae_training'):
            vae_scalar('kl_coeff', kl_coeff)
            vae_scalar('reconst_coeff', reconst_coeff)

        with tf.name_scope('vae_loss'):
            vae_scalar('loss', self.loss)
            vae_scalar('reconst_loss', img_loss)
            vae_scalar('kl_loss', latent_loss)
            vae_scalar('regularization_loss', regularization_loss)

        with tf.name_scope('vae_latent'):
            summary_avg_min_max('mu', self.z_mu, collections=['vae'])
            summary_avg_min_max('abs_mu', tf.abs(self.z_mu), collections=['vae'])
            if params.variational:
                # noinspection PyUnboundLocalVariable
                summary_avg_min_max('sigma', sigma, collections=['vae'])
            summary_avg_min_max('z', self.z, collections=['vae'])

        with tf.name_scope('vae_obs_first_0'):
            image_summaries_rgb(obs_normalized, name='obs_first', collections=['vae'])
        with tf.name_scope('vae_obs_first_dec'):
            image_summaries_rgb(obs_dec, name='obs_first_dec', collections=['vae'])


class ExplorationThroughCompression(CuriosityModule):
    class Params:
        def __init__(self):
            self.num_latent = 64
            self.variational = False
            self.intrinsic_bonus_clip = 5.0

            self.etc_batch_size = 64

    def __init__(self, env, ph_obs, agent):
        """
        :param env
        :param ph_obs - placeholder for observations
        """
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64, name='etc_step')

        self.params = agent.params
        self.ph_obs = ph_obs

        obs_space = main_observation_space(env)

        self.vae = VAE(self.ph_obs, obs_space, agent.params, agent.total_env_steps)

        self._add_summaries()
        self.summaries = merge_summaries(collections=['vae', 'etc'])

        opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='etc_opt')
        self.train_etc = opt.minimize(self.vae.loss, global_step=self.step)

        self.curr_episode_bonuses = [0] * self.params.num_envs
        self.last_episode_bonuses = [0] * self.params.num_envs
        self.last_bonuses = deque(maxlen=1000)

    def _add_summaries(self):
        with tf.name_scope('etc'):
            cm_scalar = partial(tf.summary.scalar, collections=['etc'])
            cm_scalar('rnd_step', self.step)

    def initialize(self, session):
        pass

    def generate_bonus_rewards(self, session, observations, next_obs, actions, dones, infos):
        bonuses = session.run(self.vae.losses, feed_dict={self.ph_obs: next_obs})
        bonuses *= 0.1
        assert len(bonuses) == len(dones)

        if self.params.intrinsic_bonus_clip > 0:
            bonuses = np.clip(bonuses, a_min=-self.params.intrinsic_bonus_clip, a_max=self.params.intrinsic_bonus_clip)

        bonuses = bonuses * (1 - np.array(dones))  # don't give bonus for the last transition in the episode

        self.last_bonuses.append(bonuses[0])
        self.curr_episode_bonuses += bonuses
        for i in range(len(dones)):
            if dones[i]:
                self.last_episode_bonuses[i] = self.curr_episode_bonuses[i]
                self.curr_episode_bonuses[i] = 0

        return bonuses

    def train(self, buffer, env_steps, agent):
        """Actually do a single iteration of training."""

        step = self.step.eval(session=agent.session)
        batch_size = self.params.etc_batch_size
        summary = None

        for i in range(0, len(buffer), batch_size):
            # noinspection PyProtectedMember
            with_summaries = agent._should_write_summaries(step) and summary is None
            summaries = [self.summaries] if with_summaries else []

            start, end = i, i + batch_size

            feed_dict = {self.ph_obs: buffer.obs[start:end]}

            result = agent.session.run(
                [self.train_etc] + summaries, feed_dict=feed_dict,
            )

            step += 1

            if with_summaries:
                summary = result[1]
                agent.summary_writer.add_summary(summary, global_step=env_steps)

    def set_trajectory_buffer(self, trajectory_buffer):
        """Don't need full trajectories in ETC."""
        pass

    def is_initialized(self):
        return True

    def additional_summaries(self, env_steps, summary_writer, stats_episodes, **kwargs):
        summary = tf.Summary()
        section = 'etc_aux_summaries'

        def curiosity_summary(tag, value):
            summary.value.add(tag=f'{section}/{tag}', simple_value=float(value))

        curiosity_summary('avg_episode_bonus', np.mean(self.last_episode_bonuses))
        curiosity_summary('avg_step_bonus', np.mean(self.last_bonuses))
        curiosity_summary('max_step_bonus', np.max(self.last_bonuses))
        curiosity_summary('min_step_bonus', np.min(self.last_bonuses))

        summary_writer.add_summary(summary, env_steps)

