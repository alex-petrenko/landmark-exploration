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


class NeighborhoodEncoderDeepSets(Encoder):
    """
    Using idea from "DeepSets" paper: compute a feature vector for every neighbor and aggregate them using
    an order-invariant function (mean here).
    """
    def __init__(self, neighbors, num_neighbors, params, name):
        super(NeighborhoodEncoderDeepSets, self).__init__(regularizer=None, name=name)

        with tf.variable_scope(self.name):

            # convert neighborhoods from [?, obs_shape] to [?, max_neighborhood_size, obs_shape]
            obs_shape = tf_shape(neighbors)[1:]
            neighbors = tf.reshape(neighbors, [-1, params.max_neighborhood_size] + obs_shape)

            mask = tf.sequence_mask(num_neighbors, params.max_neighborhood_size)
            mask = tf.cast(mask, dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=2)
            neighbors_masked = neighbors * tf.cast(mask, dtype=tf.float32)

            neighbors_aggregated = tf.reduce_mean(neighbors_masked, axis=1)
            self.encoded_neighborhoods = tf.layers.dense(neighbors_aggregated, 128, activation=tf.nn.relu)


def make_graph_encoder(neighbors, num_neighbors, params, name):
    """Maybe implement other encoders here."""
    if params.graph_enc_name == 'rnn':
        encoder = NeighborhoodEncoderRNN(neighbors, num_neighbors, params, name)
    elif params.graph_enc_name == 'deepsets':
        encoder = NeighborhoodEncoderDeepSets(neighbors, num_neighbors, params, name)
    else:
        raise NotImplementedError(f'Unknown graph encoder {params.graph_enc_name}')
    return encoder
