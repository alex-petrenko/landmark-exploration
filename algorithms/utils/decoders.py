import tensorflow as tf

from algorithms.utils.tf_utils import conv_t, dense


class DecoderCNN:
    def __init__(self, embedding, name, decoder_type='84px'):
        with tf.variable_scope(name):
            x = embedding

            # 84px images
            if decoder_type == '84px':
                x = dense(embedding, 3 * 3 * 64)

                x = tf.reshape(x, [-1, 3, 3, 64])
                x = conv_t(x, 64, 3, strides=1, padding='VALID')
                x = conv_t(x, 64, 3, strides=2, padding='SAME')
                x = conv_t(x, 64, 3, strides=2, padding='VALID')
                x = conv_t(x, 32, 3, strides=2, padding='SAME')
                x = conv_t(x, 1, 3, strides=2, padding='SAME', activation=tf.nn.tanh)  # [-1, 1]

            # 64px images
            elif decoder_type == '64px':
                x = tf.reshape(x, [-1, 4, 4, 64])
                x = conv_t(x, 64, 3, strides=2)
                x = conv_t(x, 64, 3, strides=2)
                x = conv_t(x, 64, 3, strides=2)
                x = conv_t(x, 1, 3, strides=2, activation=tf.nn.tanh)  # [-1, 1]

            self.decoded = x


def make_decoder(embedding, name):
    decoder = DecoderCNN(embedding, name)
    return decoder
