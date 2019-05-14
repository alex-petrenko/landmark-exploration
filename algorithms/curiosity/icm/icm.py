from functools import partial

import tensorflow as tf

from algorithms.curiosity.curiosity_module import CuriosityModule
from algorithms.utils.encoders import make_encoder, get_enc_params
from algorithms.utils.env_wrappers import main_observation_space
from algorithms.utils.tf_utils import dense, merge_summaries
from utils.utils import log, AttrDict


class IntrinsicCuriosityModule(CuriosityModule):
    """Prediction-based intrinsic curiosity."""

    class Params:
        def __init__(self):
            self.cm_beta = 0.5
            self.cm_lr_scale = 10.0
            self.clip_bonus = 0.1
            self.prediction_bonus_coeff = 0.05  # scaling factor for prediction bonus vs env rewards
            self.forward_fc = 256

    def __init__(self, env, ph_obs, ph_next_obs, ph_actions, forward_fc=256, params=None):
        """
        :param ph_obs - placeholder for observations
        :param ph_actions - placeholder for selected actions
        """
        with tf.variable_scope('icm'):
            self.params = params

            self.ph_obs = ph_obs
            self.ph_next_obs = ph_next_obs
            self.ph_actions = ph_actions

            reg = None  # don't use regularization

            obs_space = main_observation_space(env)
            num_actions = env.action_space.n

            enc_params = get_enc_params(params, 'icm_enc')
            encoder_template = tf.make_template(
                'obs_encoder', make_encoder, create_scope_now_=True,
                obs_space=obs_space, regularizer=reg, enc_params=enc_params,
            )

            encoder_obs = encoder_template(ph_obs)
            encoder_next_obs = encoder_template(ph_next_obs)

            encoded_obs = encoder_obs.encoded_input
            self.encoded_next_obs = encoder_next_obs.encoded_input

            self.feature_vector_size = encoded_obs.get_shape().as_list()[-1]
            log.info('Feature vector size in ICM/RND module: %d', self.feature_vector_size)

            actions_one_hot = tf.one_hot(ph_actions, num_actions)

            # forward model
            forward_model_input = tf.concat([encoded_obs, actions_one_hot], axis=1)
            forward_model_hidden = dense(forward_model_input, forward_fc, reg)
            forward_model_hidden = dense(forward_model_hidden, forward_fc, reg)
            forward_model_output = tf.contrib.layers.fully_connected(
                forward_model_hidden, self.feature_vector_size, activation_fn=None,
            )
            self.predicted_features = forward_model_output

            # inverse model
            inverse_model_input = tf.concat([encoded_obs, self.encoded_next_obs], axis=1)
            inverse_model_hidden = dense(inverse_model_input, 256, reg)
            inverse_model_output = tf.contrib.layers.fully_connected(
                inverse_model_hidden, num_actions, activation_fn=None,
            )
            self.predicted_actions = inverse_model_output

            self.objectives = self._objectives()

            self._add_summaries()
            self.summaries = merge_summaries(collections=['icm'])

            self.step = tf.Variable(0, trainable=False, dtype=tf.int64, name='icm_step')

            opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='icm_opt')
            self.train_icm = opt.minimize(self.objectives.loss, global_step=self.step)

    def _objectives(self):
        # model losses
        forward_loss_batch = 0.5 * tf.square(self.encoded_next_obs - self.predicted_features)
        forward_loss_batch = tf.reduce_mean(forward_loss_batch, axis=1) * self.feature_vector_size
        forward_loss = tf.reduce_mean(forward_loss_batch)

        bonus = self.params.prediction_bonus_coeff * forward_loss_batch
        self.prediction_curiosity_bonus = tf.clip_by_value(bonus, -self.params.clip_bonus, self.params.clip_bonus)

        inverse_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.predicted_actions, labels=self.ph_actions,
        ))

        cm_beta = self.params.cm_beta
        loss = forward_loss * cm_beta + inverse_loss * (1.0 - cm_beta)
        loss = self.params.cm_lr_scale * loss
        return AttrDict(locals())

    def _add_summaries(self):
        obj = self.objectives
        with tf.name_scope('losses'):
            cm_scalar = partial(tf.summary.scalar, collections=['icm'])
            cm_scalar('curiosity_forward_loss', obj.forward_loss)
            cm_scalar('curiosity_inverse_loss', obj.inverse_loss)
            cm_scalar('curiosity_module_loss', obj.loss)

    def initialize(self, session):
        pass

    def generate_bonus_rewards(self, session, observations, next_obs, actions, dones, infos):
        bonuses = session.run(
            self.prediction_curiosity_bonus,
            feed_dict={
                self.ph_actions: actions,
                self.ph_obs: observations,
                self.ph_next_obs: next_obs,
            }
        )

        bonuses = bonuses * dones  # don't give bonus for the last transition in the episode
        return bonuses

    # noinspection PyProtectedMember
    def train(self, buffer, env_steps, agent):
        """
        Actually do a single iteration of training. See the computational graph in the ctor to figure out
        the details.
        """
        step = self.step.eval(session=agent.session)
        summary = None

        for i in range(0, len(buffer), self.params.batch_size):
            with_summaries = agent._should_write_summaries(step) and summary is None
            summaries = [self.summaries] if with_summaries else []

            start, end = i, i + self.params.batch_size

            feed_dict = {
                self.ph_obs: buffer.obs[start:end],
                self.ph_next_obs: buffer.next_obs[start:end],
                self.ph_actions: buffer.actions[start:end],
            }

            result = agent.session.run(
                [self.train_icm] + summaries,
                feed_dict=feed_dict,
            )

            if with_summaries:
                summary = result[1]
                agent.summary_writer.add_summary(summary, global_step=env_steps)

    def set_trajectory_buffer(self, trajectory_buffer):
        """Don't need full trajectories in ICM."""
        pass

    def is_initialized(self):
        return True

    def additional_summaries(self, env_steps, summary_writer, stats_episodes, **kwargs):
        pass
