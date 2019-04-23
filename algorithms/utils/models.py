"""
Tensorflow models used in different RL algorithms.

- feed forward
- recurrent? (not done)

"""

import tensorflow as tf

from algorithms.utils.tf_utils import dense


class Model:
    def __init__(self, regularizer, name):
        self.name = name
        self._regularizer = regularizer


class ModelFeedForward(Model):
    def __init__(self, encoded_input, regularizer, params, name):
        super(ModelFeedForward, self).__init__(regularizer, name)
        fc_layers = params.model_fc_layers

        with tf.variable_scope(self.name):
            fc = encoded_input
            for _ in range(fc_layers):
                fc = dense(fc, params.model_fc_size, self._regularizer)

        self.latent = fc


def make_model(encoded_input, regularizer, params, name):
    model = ModelFeedForward(encoded_input, regularizer, params, name)
    return model
