import numpy as np
import tensorflow as tf
from ops import static_lstm


class Sequence_Encoder(object):
    """ An architecture that is able to encode a sequence of state feature to a feature vector """
    def __init__(self, name, is_train, config):
        self.name = name
        self._is_train = is_train
        self._reuse = False
        self.config = config

    def __call__(self, input):
        reuse = self._reuse
        with tf.variable_scope(self.name, reuse=reuse):
            if not reuse:
                print('\033[93m'+self.name+'\033[0m')

            output, state = static_lstm(input, self.config.lstm_latent_size, info=not reuse, name='encoder_lstm')

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return output, state
