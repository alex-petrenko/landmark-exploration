import numpy as np
import tensorflow as tf
from ops import fc
from ops import bilinear_deconv2d as deconv2d


class State_Decoder(object):
    """ An architecture that is able to decode a feature vector to the output space """
    def __init__(self, name, is_train, config):
        self.name = name
        self._is_train = is_train
        self._reuse = False
        self.config = config

    def __call__(self, input, output_activation):
        raise NotImplementedError


class State_Decoder_Fc(State_Decoder):
    def __init__(self, name, is_train, config, output_size):
        super(State_Decoder_Fc, self).__init__(name, is_train, config)
        self.output_size = 1 if not output_size else output_size[0]

    def __call__(self, input_list, output_activation):
        input_, _ = input_list      # second argument (skip values) not used for FC network
        reuse = self._reuse
        is_train = self._is_train
        with tf.variable_scope(self.name, reuse=reuse):
            if not reuse:
                print('\033[93m' + self.name + '\033[0m')
            _ = input_

            for i, num_layer_outputs in enumerate(self.config.enc_dec_layers[::-1]):
                _ = fc(_, num_layer_outputs,
                       is_train, info=not reuse, name='fc{}'.format(i + 1))
            # final output with given activation function
            _ = fc(_, self.output_size, is_train, activation_fn=output_activation,
                   info=not reuse, name='fc{}'.format(i + 2))

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return _


class State_Decoder_Conv(State_Decoder):
    def __init__(self, name, is_train, config, output_channels):
        super(State_Decoder_Conv, self).__init__(name, is_train, config)
        self.output_channels = output_channels[-1]

    def __call__(self, input_list, output_activation):
        input_, skips = input_list
        reuse = self._reuse
        is_train = self._is_train
        with tf.variable_scope(self.name, reuse=reuse):
            if not reuse:
                print('\033[93m' + self.name + '\033[0m')
            _ = input_

            # compute required initial resolution
            batch_size, latent_size = input_.get_shape().as_list()
            initial_res = self.config.resolution / (2 ** (len(self.config.enc_dec_layers)+1))

            # lift latent to initial spatial resolution
            _ = fc(_, initial_res**2,
                   is_train, activation_fn=None, info=not reuse, name='fc_0')
            _ = tf.reshape(_, (batch_size, initial_res, initial_res, 1))

            for i, (output_channels, skip) in enumerate(zip(self.config.enc_dec_layers[::-1], skips[::-1])):
                if self.config.use_skip_connections:
                    concat_input = tf.concat((_, skip), axis=-1)
                else:
                    concat_input = _
                # deconv with fixed kernel size 4x4 and stride 2x2
                _ = deconv2d(concat_input, [output_channels, 4, 2], is_train, info=not reuse, name='deconv{}'.format(i + 2))
            if self.config.use_skip_connections:
                concat_input = tf.concat((_, skips[0]), axis=-1)
            else:
                concat_input = _
            _ = deconv2d(concat_input, [self.output_channels, 4, 2], is_train, activation_fn=output_activation,
                         info=not reuse, norm='None', name='deconv{}'.format(i + 3))

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return _


def get_state_decoder(name, is_train, config, output_size):
    dataset = config.dataset
    if dataset == 'sin' or not config.image_input:
        return State_Decoder_Fc(name, is_train, config, output_size)
    elif dataset == "bouncing_balls":
        return State_Decoder_Conv(name, is_train, config, output_size)
    else:
        raise NotImplementedError
