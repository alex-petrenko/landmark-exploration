import tensorflow as tf

from algorithms.encoders import make_encoder
from algorithms.env_wrappers import get_observation_space
from algorithms.tf_utils import dense, placeholders_from_spaces


class ReachabilityNetwork:
    def __init__(self, env, params):
        self.ph_ob1, self.ph_ob2 = placeholders_from_spaces(get_observation_space(env), get_observation_space(env))
        self.ph_labels = tf.placeholder(dtype=tf.int32, shape=(None, ))

        with tf.variable_scope('reach'):

            def make_encoder_cnn(ph_obs):
                enc = make_encoder(env, ph_obs, None, params, 'reach_enc')
                return enc.encoded_input

            encoder = tf.make_template('siamese_enc', make_encoder_cnn)

            ob1_enc = encoder(self.ph_ob1)
            ob2_enc = encoder(self.ph_ob2)
            observations_encoded = tf.concat([ob1_enc, ob2_enc], axis=1)

            fc_layers = [256, 256]
            x = observations_encoded
            for fc_layer_size in fc_layers:
                x = dense(x, fc_layer_size)

            logits = tf.contrib.layers.fully_connected(x, 2, activation_fn=None)
            self.probabilities = tf.nn.softmax(logits)

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.ph_labels)

    def get_probabilities(self, session, ob1, ob2):
        probabilities = session.run(
            self.probabilities,
            feed_dict={self.ph_ob1: ob1, self.ph_ob2: ob2},
        )
        return probabilities
