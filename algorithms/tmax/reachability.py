import tensorflow as tf

from algorithms.encoders import make_encoder
from algorithms.env_wrappers import get_observation_space
from algorithms.tf_utils import dense, placeholders_from_spaces


class ReachabilityNetwork:
    def __init__(self, env, params):
        obs_space = get_observation_space(env)
        self.ph_obs_first, self.ph_obs_second = placeholders_from_spaces(obs_space, obs_space)
        self.ph_labels = tf.placeholder(dtype=tf.int32, shape=(None, ))

        with tf.variable_scope('reach'):
            def make_encoder_cnn(ph_obs):
                enc = make_encoder(env, ph_obs, None, params, 'reach_enc')
                return enc.encoded_input

            encoder = tf.make_template('siamese_enc', make_encoder_cnn)

            ob1_enc = encoder(self.ph_obs_first)
            ob2_enc = encoder(self.ph_obs_second)
            observations_encoded = tf.concat([ob1_enc, ob2_enc], axis=1)

            fc_layers = [256, 256]
            x = observations_encoded
            for fc_layer_size in fc_layers:
                x = dense(x, fc_layer_size)

            logits = tf.contrib.layers.fully_connected(x, 2, activation_fn=None)
            self.probabilities = tf.nn.softmax(logits)

            self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.ph_labels)
            self.loss = tf.reduce_mean(self.loss)

    def get_probabilities(self, session, obs_first, obs_second):
        probabilities = session.run(
            self.probabilities,
            feed_dict={self.ph_obs_first: obs_first, self.ph_obs_second: obs_second},
        )
        return probabilities

    def get_reachability(self, session, obs_first, obs_second):
        probs = self.get_probabilities(session, obs_first, obs_second)
        return [p[1] for p in probs]

