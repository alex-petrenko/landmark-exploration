import tensorflow as tf

from algorithms.encoders import Encoder
from algorithms.tf_utils import tf_shape


class NeighborhoodEncoderRNN(Encoder):
    """
    Encode the graph neighborhood of the observation.
    This particular implementation processes neighbor vertices (0 or 1 edge away in the graph) as sequence.
    """

    def __init__(self, neighbors, num_neighbors, params, name):
        """
        Ctor.
        :param neighbors: observations that are close to the current observation in the topological sense (encoded)
        :param num_neighbors: tensor, lenghts of sequences (sizes of neighborhoods)
        """
        super(NeighborhoodEncoderRNN, self).__init__(regularizer=None, name=name)

        with tf.variable_scope(self.name):

            # convert neighborhoods from [?, obs_shape] to [?, max_neighborhood_size, obs_shape]
            obs_shape = tf_shape(neighbors)[1:]
            neighbors = tf.reshape(neighbors, [-1, params.max_neighborhood_size] + obs_shape)

            cell = tf.nn.rnn_cell.GRUCell(params.graph_encoder_rnn_size)
            # noinspection PyUnresolvedReferences
            _, last_states = tf.nn.dynamic_rnn(
                cell=cell, inputs=neighbors, sequence_length=num_neighbors, dtype=tf.float32,
            )

            # last states of the dynamic GRU will contain the internal state (which is the same as the output of the
            # cell) for the last item in the sequence. E.g. if len=3 then only 3 RNN steps (0th, 1st and 2nd) will
            # be calculated and last_states will contain the output of the 2nd step, regardless of max horizon
            self.encoded_neighborhoods = tf.layers.dense(
                last_states, params.graph_encoder_rnn_size, activation=tf.nn.relu,
            )


def make_graph_encoder(neighbors, num_neighbors, params, name):
    """Maybe implement other encoders here."""
    encoder = NeighborhoodEncoderRNN(neighbors, num_neighbors, params, name)
    return encoder
