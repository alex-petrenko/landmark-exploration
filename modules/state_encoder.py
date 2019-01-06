import numpy as np
import tensorflow as tf
from ops import fc, conv2d


class State_Encoder(object):
    """ An architecture that is able to encode a raw state to a feature vector """
    def __init__(self, name, is_train, config):
        self.name = name
        self._is_train = is_train
        self._reuse = False
        self.config = config

    def __call__(self, input):
        raise NotImplementedError


class State_Encoder_Fc(State_Encoder):
    def __init__(self, name, is_train, config):
        super(State_Encoder_Fc, self).__init__(name, is_train, config)

    def __call__(self, input):
        reuse = self._reuse
        is_train = self._is_train
        with tf.variable_scope(self.name, reuse=reuse):
            if not reuse:
                print('\033[93m' + self.name + '\033[0m')
            _ = input

            for i, output_neurons in enumerate(self.config.enc_dec_layers):
                _ = fc(_, output_neurons, is_train, info=not reuse, name='fc{}'.format(i + 1))
            _ = fc(_, self.config.enc_latent_size,
                   is_train, activation_fn=None, info=not reuse, name='fc{}'.format(i + 2))

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return _, None      # no skip connection output for FC networks


class State_Encoder_Conv(State_Encoder):
    def __init__(self, name, is_train, config):
        super(State_Encoder_Conv, self).__init__(name, is_train, config)

    def __call__(self, input):
        reuse = self._reuse
        is_train = self._is_train
        with tf.variable_scope(self.name, reuse=reuse):
            if not reuse:
                print('\033[93m' + self.name + '\033[0m')
            _ = input

            # check input dims
            batch_size, width, height, channels = input.get_shape().as_list()
            assert width == height, "Only implemented ConvStateEncoder for squared input"

            skip_outputs = []
            for i, output_channels in enumerate(self.config.enc_dec_layers):
                _ = conv2d(_, output_channels, is_train, info=not reuse, name='conv{}'.format(i + 1))
                skip_outputs.append(_)
            _ = conv2d(_, 1, is_train, activation_fn=None, info=not reuse, name='conv{}'.format(i + 2))
            skip_outputs.append(_)

            # flatten the final output
            _ = tf.reshape(_, [batch_size, -1])
            if _.get_shape().as_list()[-1] != self.config.enc_latent_size:
                print('\033[93m' + "ConvStateEncoder output_size of %d does not match specified %d, "
                                   "applying FC layer to correct!" %
                      (_.get_shape().as_list()[-1], self.config.enc_latent_size) + '\033[0m')
                _ = fc(_, self.config.enc_latent_size,
                       is_train, activation_fn=None, info=not reuse, name='fc{}'.format(i + 3))

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return _, skip_outputs


def get_state_encoder(name, is_train, config):
    dataset = config.dataset
    if dataset == "sin" or not config.image_input:
        return State_Encoder_Fc(name, is_train, config)
    elif dataset == "bouncing_balls":
        return State_Encoder_Conv(name, is_train, config)
    else:
        raise NotImplementedError