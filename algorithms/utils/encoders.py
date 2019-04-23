"""
Tensorflow encoders used in different trainable models.

"""
import numpy as np
import tensorflow as tf

from algorithms.utils.env_wrappers import has_image_observations
from algorithms.utils.tf_utils import dense, conv, put_kernels_on_grid, tf_shape


class EncoderParams:
    def __init__(self):
        self.enc_name = None

        self.batch_norm = False
        self.ph_is_training = None

        self.summary_collections = None


def get_enc_params(params, summary_collection=None):
    enc_params = EncoderParams()
    enc_params.enc_name = params.image_enc_name
    enc_params.summary_collections = [summary_collection]
    return enc_params


class Encoder:
    """A part of the graph responsible for the input encoding."""

    def __init__(self, regularizer, name):
        self.name = name
        self._regularizer = regularizer


class EncoderCNN(Encoder):
    def __init__(self, normalized_obs, regularizer, enc_params, name):
        super(EncoderCNN, self).__init__(regularizer, name)

        self.normalized_obs = normalized_obs
        self.enc_params = enc_params

        img_enc_name = enc_params.enc_name

        with tf.variable_scope(self.name):
            if img_enc_name == 'convnet_simple':
                conv_filters = self._convnet_simple([(16, 5, 2), (32, 3, 2), (32, 3, 2), (64, 3, 2)])
            elif img_enc_name == 'convnet_42px':
                conv_filters = self._convnet_simple([(32, 3, 2)] * 4)
            elif img_enc_name == 'convnet_84px':
                conv_filters = self._convnet_simple([(16, 3, 2)] + [(32, 3, 2)] * 4)
            elif img_enc_name == 'convnet_84px_8x8':
                conv_filters = self._convnet_simple([(32, 8, 4), (32, 4, 2), (32, 3, 2), (32, 3, 2)])
            else:
                raise Exception('Unknown model name')

            self.encoded_w, self.encoded_h, self.encoded_channels = tf_shape(conv_filters)[1:]
            self.encoded_input = tf.contrib.layers.flatten(conv_filters)

            # summaries
            with tf.variable_scope('conv1', reuse=True):
                weights = tf.get_variable('weights')
            with tf.name_scope('summary_conv'):
                if weights.shape[2].value in [1, 3, 4]:
                    tf.summary.image(
                        'conv1/kernels',
                        put_kernels_on_grid(weights),
                        max_outputs=1,
                        collections=enc_params.summary_collections,
                    )

    def _conv(self, x, filters, kernel, stride, scope=None):
        return conv(
            x,
            filters,
            kernel,
            stride=stride,
            regularizer=self._regularizer,
            batch_norm=self.enc_params.batch_norm,
            is_training=self.enc_params.ph_is_training,
            scope=scope,
        )

    def _convnet_simple(self, convs):
        """Basic stacked convnet."""
        layer = self.normalized_obs
        layer_idx = 1
        for filters, kernel, stride in convs:
            layer = self._conv(layer, filters, kernel, stride, 'conv' + str(layer_idx))
            layer_idx += 1
        return layer


def is_normalized(obs_space):
    return obs_space.dtype == np.float32 and obs_space.low in [-1.0, 0.0] and obs_space.high == 1.0


def tf_normalize(obs, obs_space):
    """Result will be float32 tensor with values in [-1, 1]."""
    low, high = obs_space.low.flat[0], obs_space.high.flat[0]
    mean = (low + high) * 0.5
    if obs_space.dtype != np.float32:
        obs = tf.to_float(obs)

    scaling = 1.0 / (high - mean)
    obs = (obs - mean) * scaling
    return obs


def make_encoder(ph_observations, obs_space, regularizer, enc_params, name='enc'):
    """
    Create an appropriate encoder according to params.
    'name' argument is used to create an internal variable scope, which allows you to use this function to create
    multiple encoders (without parameter sharing) by providing a different name.
    If you're sharing encoder parameters, use tf.make_template, and make sure that 'name' stays default or passed as
    a keyword argument when the encoder template as created (so it's the same for all instances of the shared encoder)

    :param ph_observations: observation placeholder
    :param obs_space: environment observation space
    :param regularizer: reg
    :param enc_params: encoder params (see EncoderParams)
    :param name: used to create an internal variable scope! Be careful!  <---
    """
    if has_image_observations(obs_space):
        if is_normalized(obs_space):
            obs_normalized = ph_observations
        else:
            obs_normalized = tf_normalize(ph_observations, obs_space)

        encoder = EncoderCNN(obs_normalized, regularizer, enc_params, name)
    else:
        raise NotImplementedError

    return encoder


class EncoderWithGoal:
    def __init__(self, ph_observations, ph_goal_obs, obs_space, reg, enc_params, name):
        with tf.variable_scope(name):
            enc_template = tf.make_template(
                'obs_enc', make_encoder,
                create_scope_now_=True, obs_space=obs_space, regularizer=reg, enc_params=enc_params,
            )

            # obs and goal encoders share parameters (via make_template)
            self.encoder_obs = enc_template(ph_observations)
            self.encoder_goal = enc_template(ph_goal_obs)

            self.encoded_input = tf.concat([self.encoder_obs.encoded_input, self.encoder_goal.encoded_input], axis=1)

            # obs_concat = tf.concat([ph_observations, ph_goal_obs], axis=3)
            # self.encoder_obs = self.encoded_input = enc_template(obs_concat).encoded_input


def make_encoder_with_goal(ph_observations, ph_goal_obs, obs_space, regularizer, enc_params, name='enc_with_goal'):
    main_obs_space = goal_obs_space = obs_space

    # main and goal obs spaces should be the same
    assert main_obs_space.low.flat[0] == goal_obs_space.low.flat[0]
    assert main_obs_space.high.flat[0] == goal_obs_space.high.flat[0]

    encoder_with_goal = EncoderWithGoal(ph_observations, ph_goal_obs, main_obs_space, regularizer, enc_params, name)
    return encoder_with_goal


