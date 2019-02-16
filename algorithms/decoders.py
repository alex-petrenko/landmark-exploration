import tensorflow as tf

from algorithms.tf_utils import conv_t


class DecoderCNN:
    def __init__(self, embedding, name):
        with tf.variable_scope(name):
            self.decoded = self._stacked_deconvs(embedding, [(32, 3, 2)] * 4 + [(16, 3, 2)])

    @staticmethod
    def _stacked_deconvs(x, deconvs):
        """Basic stacked deconvnet."""
        for filters, kernel, stride in deconvs:
            x = conv_t(x, filters, kernel, stride)
        return x
