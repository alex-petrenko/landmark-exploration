from functools import partial

import numpy as np
import tensorflow as tf

from algorithms.curiosity.curiosity_module import CuriosityModule
from algorithms.utils.encoders import make_encoder, get_enc_params
from algorithms.utils.env_wrappers import main_observation_space
from algorithms.utils.tf_utils import dense, merge_summaries
from utils.utils import log, AttrDict


class RandomNetworkDistillation(CuriosityModule):
    """Prediction-based intrinsic curiosity, tries to predict output of a random but fixed network on an observation"""

    class Params:  # TODO: Remove what you don't need
        def __init__(self):
            self.prediction_loss_scale = 10.0
            self.intrinsic_bonus_clip = 0.1
            if self.intrinsic_bonus_clip:
                self.intrinsic_bonus_min = - self.intrinsic_bonus_clip
                self.intrinsic_bonus_max = self.intrinsic_bonus_clip
            self.prediction_bonus_coeff = 0.01  # scaling factor for prediction bonus vs env rewards
            self.forward_fc = 256

    def __init__(self, env, ph_obs, params=None):
        """
        :param env
        :param ph_obs - placeholder for observations
        """
        with tf.variable_scope('rnd'):
            self.params = params
            self.ph_obs = ph_obs

            reg = None  # don't use regularization

            obs_space = main_observation_space(env)

            target_enc_params = get_enc_params(params, 'rnd_target')
            encoder_obs = make_encoder(ph_obs, obs_space, reg, target_enc_params, name='target_encoder')
            self.predicted_features = encoder_obs.encoded_input

            predictor_enc_params = get_enc_params(params, 'rnd_predictor')
            target_features = make_encoder(ph_obs, obs_space, reg, predictor_enc_params, name='predictor_encoder')
            self.tgt_features = tf.stop_gradient(target_features.encoded_input)

            self.feature_vector_size = self.predicted_features.get_shape().as_list()[-1]
            log.info('Feature vector size in RND module: %d', self.feature_vector_size)

            self.objectives = self._objectives()

            self._add_summaries()
            self.summaries = merge_summaries(collections=['rnd'])

            self.step = tf.Variable(0, trainable=False, dtype=tf.int64, name='rnd_step')

            opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='rnd_opt')
            self.train_rnd = opt.minimize(self.objectives.loss, global_step=self.step)

    def _objectives(self):
        # model losses
        l2_loss_obs = tf.nn.l2_loss(self.tgt_features - self.predicted_features)
        #one axis must have dimension None
        prediction_loss = tf.reduce_mean(l2_loss_obs, axis=0)
        bonus = prediction_loss

        loss = prediction_loss * self.params.prediction_loss_scale
        return AttrDict(locals())

    def _add_summaries(self):
        obj = self.objectives
        with tf.name_scope('losses'):
            cm_scalar = partial(tf.summary.scalar, collections=['rnd'])
            cm_scalar('rnd_loss', obj.loss)

    def initialize(self, session):
        pass

    def generate_bonus_rewards(self, session, observations, next_obs, actions, dones, infos):
        bonuses = session.run( #TODO: bonuses should not be scalar
            self.objectives.bonus,
            feed_dict={
                self.ph_obs: observations,
            }
        )
        print(bonuses[:4])
        bonuses *= self.params.prediction_bonus_coeff # TODO: coeff is not scaling right: check paper
        if self.params.intrinsic_bonus_clip:
            bonuses = np.clip(bonuses, a_min=self.params.intrinsic_bonus_min, a_max=self.params.intrinsic_bonus_max)

        bonuses = bonuses * (1 - np.array(dones))  # don't give bonus for the last transition in the episode
        print(bonuses[:4])
        print(dones[:4])
        return bonuses

    def train(self, buffer, env_steps, agent):
        """
        Actually do a single iteration of training
        """
        step = self.step.eval(session=agent.session)
        summary = None

        for i in range(0, len(buffer), self.params.batch_size):
            with_summaries = agent._should_write_summaries(step) and summary is None
            summaries = [self.summaries] if with_summaries else []

            start, end = i, i + self.params.batch_size

            feed_dict = {
                self.ph_obs: buffer.obs[start:end],
            }

            result = agent.session.run(
                [self.train_rnd] + summaries,
                feed_dict=feed_dict,
            )

            if with_summaries:
                summary = result[1]
                agent.summary_writer.add_summary(summary, global_step=env_steps)

    def set_trajectory_buffer(self, trajectory_buffer):
        """Don't need full trajectories in RND."""
        pass

    def is_initialized(self):
        return True

    def additional_summaries(self, env_steps, summary_writer, stats_episodes, **kwargs):
        pass
