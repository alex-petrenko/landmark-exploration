"""
Tensorflow encoders used in different trainable models.

"""

import tensorflow as tf

from algorithms.env_wrappers import has_image_observations
from algorithms.tf_utils import dense, conv
from utils.utils import put_kernels_on_grid


class Encoder:
    """A part of the graph responsible for the input encoding."""

    def __init__(self, regularizer, params, name):
        self.name = name

        self._past_frames = params.stack_past_frames
        self._num_frames = params.num_input_frames

        self._regularizer = regularizer


class EncoderCNN(Encoder):
    def __init__(self, ph_observations, regularizer, params, name):
        super(EncoderCNN, self).__init__(regularizer, params, name)

        self._ph_observations = ph_observations
        img_enc_name = params.image_enc_name

        with tf.variable_scope(self.name):
            if img_enc_name == 'convnet_simple':
                conv_filters = self._convnet_simple([(16, 5, 2), (32, 3, 2), (32, 3, 2), (64, 3, 2)])
            elif img_enc_name == 'convnet_large':
                conv_filters = self._convnet_simple([(32, 5, 2), (64, 3, 2), (128, 3, 2), (128, 3, 2), (128, 3, 2)])
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
    def __init__(self, env, ph_observations, regularizer, params, name):
        super(EncoderLowDimensional, self).__init__(regularizer, params, name)

        lowdim_enc_name = params.lowdim_enc_name

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


def make_encoder(env, ph_observations, regularizer, params, name):
    if has_image_observations(env.observation_space.spaces['obs']):
        encoder = EncoderCNN(ph_observations, regularizer, params, name)
    else:
        encoder = EncoderLowDimensional(env, ph_observations, regularizer, params, name)
    return encoder
