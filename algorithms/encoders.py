"""
Tensorflow encoders used in different trainable models.

"""
import numpy as np

import tensorflow as tf

from algorithms.env_wrappers import has_image_observations, get_observation_space
from algorithms.tf_utils import dense, conv
from utils.utils import put_kernels_on_grid


class Encoder:
    """A part of the graph responsible for the input encoding."""

    def __init__(self, regularizer, name):
        self.name = name
        self._regularizer = regularizer


class EncoderCNN(Encoder):
    def __init__(self, ph_observations, regularizer, img_enc_name, name):
        super(EncoderCNN, self).__init__(regularizer, name)

        self._ph_observations = ph_observations

        with tf.variable_scope(self.name):
            if img_enc_name == 'convnet_simple':
                conv_filters = self._convnet_simple([(16, 5, 2), (32, 3, 2), (32, 3, 2), (64, 3, 2)])
            elif img_enc_name == 'convnet_doom_small':
                conv_filters = self._convnet_simple([(32, 3, 2)] * 4)  # to fairly compare with previous algos
            elif img_enc_name == 'convnet_doom':
                # use this with 64x64 resolution
                conv_filters = self._convnet_simple([(32, 5, 2), (32, 3, 2), (32, 3, 2), (64, 3, 2), (64, 3, 2)])
            else:
                raise Exception('Unknown model name')

            self.encoded_input = tf.contrib.layers.flatten(conv_filters)

            # summaries
            with tf.variable_scope('conv1', reuse=True):
                weights = tf.get_variable('weights')
            with tf.name_scope('summary_conv'):
                if weights.shape[2].value in [1, 3, 4]:
                    # TODO: add summary to collection?
                    tf.summary.image('conv1/kernels', put_kernels_on_grid(weights), max_outputs=1)

    def _conv(self, x, filters, kernel, stride, scope=None):
        return conv(x, filters, kernel, stride=stride, regularizer=self._regularizer, scope=scope)

    def _convnet_simple(self, convs):
        """Basic stacked convnet."""
        layer = self._ph_observations
        layer_idx = 1
        for filters, kernel, stride in convs:
            layer = self._conv(layer, filters, kernel, stride, 'conv' + str(layer_idx))
            layer_idx += 1
        return layer


class EncoderLowDimensional(Encoder):
    def __init__(self, ph_observations, regularizer, lowdim_enc_name, name):
        super(EncoderLowDimensional, self).__init__(regularizer, name)

        with tf.variable_scope(self.name):
            if lowdim_enc_name == 'simple_fc':
                frames = tf.split(ph_observations, self._num_frames, axis=1)
                fc_encoder = tf.make_template(
                    'fc_enc_' + self.name, self._fc_frame_encoder, create_scope_now_=True
                )
                encoded_frames = [fc_encoder(frame) for frame in frames]
                self.encoded_input = tf.concat(encoded_frames, axis=1)
            else:
                raise Exception('Unknown lowdim model name')

    def _fc_frame_encoder(self, x):
        return dense(x, 128, self._regularizer)


def is_normalized(obs_space):
    return obs_space.dtype == np.float32 and obs_space.low in [0.0, 1.0] and obs_space.high == 1.0


def tf_normalize(obs, obs_space):
    low, high = obs_space.low, obs_space.high
    mean = (low + high) * 0.5
    if obs_space.dtype != np.float32:
        obs = tf.to_float(obs)
    obs = (obs - mean) / (high - mean)
    return obs


def make_encoder(env, ph_observations, regularizer, params, name):
    obs_space = get_observation_space(env)
    if has_image_observations(obs_space):
        if is_normalized(obs_space):
            obs_normalized = ph_observations
        else:
            obs_normalized = tf_normalize(ph_observations, obs_space)

        encoder = EncoderCNN(obs_normalized, regularizer, params.image_enc_name, name)
    else:
        encoder = EncoderLowDimensional(ph_observations, regularizer, params.lowdim_enc_name, name)
    return encoder
