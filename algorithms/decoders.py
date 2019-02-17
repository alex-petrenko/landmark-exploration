import tensorflow as tf

from algorithms.tf_utils import conv_t


class DecoderCNN:
    def __init__(self, embedding, name):
        with tf.variable_scope(name):
            x = embedding

            # use that for 84px
            # x = tf.reshape(x, [-1, 2, 2, 64])
            # x = conv_t(x, 32, 3, strides=2, padding='VALID')
            # x = conv_t(x, 32, 3, strides=2, padding='SAME')
            # x = conv_t(x, 32, 3, strides=2, padding='VALID')
            # x = conv_t(x, 32, 3, strides=2, padding='SAME')
            # x = conv_t(x, 1, 3, strides=2, padding='SAME', activation=tf.nn.tanh)  # [-1, 1]

            # 64px images
            x = tf.reshape(x, [-1, 4, 4, 64])
            x = conv_t(x, 64, 3, strides=2)
            x = conv_t(x, 64, 3, strides=2)
            x = conv_t(x, 64, 3, strides=2)
            x = conv_t(x, 1, 3, strides=2, activation=tf.nn.tanh)  # [-1, 1]

            self.decoded = x
