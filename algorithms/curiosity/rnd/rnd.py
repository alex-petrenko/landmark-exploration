from functools import partial

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
            self.cm_lr_scale = 10.0
            self.clip_bonus = 0.1
            self.prediction_bonus_coeff = 0.05  # scaling factor for prediction bonus vs env rewards
            self.forward_fc = 256

    def __init__(self, env, ph_obs, forward_fc=256, params=None):
        """
        :param env
        :param ph_obs - placeholder for observations
        :param forward_fc
        """
        with tf.variable_scope('rnd'):
            self.params = params
            self.ph_obs = ph_obs

            reg = None  # don't use regularization

            obs_space = main_observation_space(env)

            target_enc_params = get_enc_params(params, 'rnd_target')  # fixed but random enc
            encoder_template = tf.make_template(
                'obs_encoder', make_encoder, create_scope_now_=True,
                obs_space=obs_space, regularizer=reg, enc_params=target_enc_params,
            )

            encoder_obs = encoder_template(ph_obs, name='encoder')
            self.predicted_features = encoder_obs.encoded_input

            tgt_encoder_obs = encoder_template(ph_obs, trainable=False, name='tgt_encoder')
            self.tgt_encoded_obs = tf.stop_gradient(tgt_encoder_obs.encoded_input)

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
        l2_loss_obs = tf.nn.l2_loss(self.tgt_encoded_obs - self.predicted_features)
        prediction_loss = tf.reduce_mean(l2_loss_obs)

        bonus = self.params.prediction_bonus_coeff * prediction_loss
        self.prediction_curiosity_bonus = tf.clip_by_value(bonus, -self.params.clip_bonus, self.params.clip_bonus)

        loss = self.params.cm_lr_scale * prediction_loss
        return AttrDict(locals())

    def _add_summaries(self):
        obj = self.objectives
        with tf.name_scope('losses'):
            cm_scalar = partial(tf.summary.scalar, collections=['rnd'])
            cm_scalar('rnd_loss', obj.loss)

    def initialize(self, session):
        pass

    def generate_bonus_rewards(self, session, observations, next_obs, actions, dones, infos):
        bonuses = session.run(
            self.prediction_curiosity_bonus,
            feed_dict={
                self.ph_obs: observations,
            }
        )

        bonuses = bonuses * dones  # don't give bonus for the last transition in the episode
        return bonuses

    # noinspection PyProtectedMember
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
