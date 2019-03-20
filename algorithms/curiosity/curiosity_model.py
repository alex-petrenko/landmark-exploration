import tensorflow as tf

from algorithms.encoders import make_encoder
from algorithms.env_wrappers import has_image_observations, main_observation_space
from algorithms.tf_utils import dense, count_total_parameters, conv
from utils.utils import log


class CuriosityModel:
    """Single class for inverse and forward dynamics model."""

    def __init__(self, env, obs, next_obs, actions, past_frames, forward_fc, params=None):
        """
        :param obs - placeholder for observations
        :param actions - placeholder for selected actions
        """
        with tf.variable_scope('curiosity_model'):
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=1e-10)

            obs_space = main_observation_space(env)
            image_obs = has_image_observations(obs_space)
            num_actions = env.action_space.n

            if image_obs:
                encoded_obs = make_encoder(obs, obs_space, self.regularizer, params, name='encoded_obs')
                encoded_next_obs = make_encoder(next_obs, obs_space, self.regularizer, params, name='encoded_next_obs')
                encoded_obs = encoded_obs.encoded_input
                self.encoded_next_obs = encoded_next_obs.encoded_input
            else:
                # low-dimensional input
                lowdim_encoder = tf.make_template(
                    'lowdim_encoder',
                    self._lowdim_encoder,
                    create_scope_now_=True,
                    past_frames=past_frames,
                )
                encoded_obs = lowdim_encoder(obs=obs)
                self.encoded_next_obs = lowdim_encoder(obs=next_obs)

            self.feature_vector_size = encoded_obs.get_shape().as_list()[-1]
            log.info('Feature vector size in ICM/RND module: %d', self.feature_vector_size)

            actions_one_hot = tf.one_hot(actions, num_actions)

            # forward model
            forward_model_input = tf.concat(
                [encoded_obs, actions_one_hot],
                axis=1,
            )
            forward_model_hidden = dense(forward_model_input, forward_fc, self.regularizer)
            forward_model_hidden = dense(forward_model_hidden, forward_fc, self.regularizer)
            forward_model_output = tf.contrib.layers.fully_connected(
                forward_model_hidden, self.feature_vector_size, activation_fn=None,
            )
            self.predicted_obs = forward_model_output

            # inverse model
            inverse_model_input = tf.concat([encoded_obs, self.encoded_next_obs], axis=1)
            inverse_model_hidden = dense(inverse_model_input, 256, self.regularizer)
            inverse_model_output = tf.contrib.layers.fully_connected(
                inverse_model_hidden, num_actions, activation_fn=None,
            )
            self.predicted_actions = inverse_model_output

            log.info('Total parameters in the model: %d', count_total_parameters())

    def _fc_frame_encoder(self, x):
        return dense(x, 128, self.regularizer)

    def _lowdim_encoder(self, obs, past_frames):
        frames = tf.split(obs, past_frames, axis=1)
        fc_encoder = tf.make_template('fc_encoder', self._fc_frame_encoder, create_scope_now_=True)
        encoded_frames = [fc_encoder(frame) for frame in frames]
        encoded_input = tf.concat(encoded_frames, axis=1)
        return encoded_input

    def _conv(self, x, filters, kernel, stride, scope=None):
        return conv(x, filters, kernel, stride=stride, regularizer=self.regularizer, scope=scope)

    def _convnet_simple(self, convs, obs):
        """Basic stacked convnet."""
        layer = obs
        layer_idx = 1
        for filters, kernel, stride in convs:
            layer = self._conv(layer, filters, kernel, stride, 'conv' + str(layer_idx))
            layer_idx += 1
        return layer
