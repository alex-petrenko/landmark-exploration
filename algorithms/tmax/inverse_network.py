import tensorflow as tf

from algorithms.buffer import Buffer
from algorithms.decoders import DecoderCNN
from algorithms.encoders import make_encoder
from algorithms.env_wrappers import get_observation_space
from algorithms.tf_utils import dense, placeholder_from_space, placeholders_from_spaces
from utils.timing import Timing
from utils.utils import log


class InverseNetwork:
    """Inverse network to predict the current action using a pair of states."""
    def __init__(self, env, params):
        obs_space = get_observation_space(env)
        self.ph_obs_first, self.ph_obs_second = placeholders_from_spaces(obs_space, obs_space)
        self.ph_actions = placeholder_from_space(env.action_space)

        with tf.variable_scope('inverse'):
            reg = tf.contrib.layers.l2_regularizer(scale=1e-4)

            encoder = tf.make_template(
                'siamese_enc_inverse', make_encoder, create_scope_now_=True, env=env, regularizer=reg, params=params,
            )

            obs_first_enc = encoder(self.ph_obs_first)
            obs_second_enc = encoder(self.ph_obs_second)
            x = tf.concat([obs_first_enc.encoded_input, obs_second_enc.encoded_input], axis=1)

            fc_layers = [256, 256]
            for fc_layer_size in fc_layers:
                x = dense(x, fc_layer_size, reg)

            action_logits = tf.layers.dense(x, env.action_space.n, activation=None)

            self.actions_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=action_logits, labels=self.ph_actions)
            self.actions_loss = tf.reduce_mean(self.actions_loss)

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.reg_loss = tf.reduce_sum(reg_losses)

            first_feature_vector = tf.stop_gradient(obs_first_enc.encoded_input)
            self.obs_decoded = DecoderCNN(first_feature_vector, 'inv_dec').decoded
            self.normalized_obs = obs_first_enc.normalized_obs
            self.reconst_loss = tf.nn.l2_loss(self.normalized_obs - self.obs_decoded) / (84 * 84)

            self.loss = self.actions_loss + self.reg_loss + self.reconst_loss


class InverseBuffer:
    """Training data for the reachability network (observation pairs and labels)."""

    def __init__(self, params):
        self.buffer = Buffer()
        self.params = params

    def extract_data(self, trajectories):
        timing = Timing()

        with timing.timeit('trajectories'):
            for trajectory in trajectories:
                if len(trajectory) <= 1:
                    continue

                obs = trajectory.obs
                act = trajectory.actions
                deliberate = trajectory.deliberate_action

                for i in range(len(trajectory) - 1):
                    if not deliberate[i]:
                        continue

                    self.buffer.add(
                        obs_i=obs[i],
                        obs_i_plus_1=obs[i + 1],
                        actions=act[i],
                        i=i,
                    )

            self.shuffle_data()
            self.buffer.trim_at(50000)

        log.info('Inverse timing %s', timing)

    def has_enough_data(self):
        len_data, min_data = len(self.buffer), 1000
        if len_data < min_data:
            log.info('Need to gather more data to train inverse net, %d/%d', len_data, min_data)
            return False
        return True

    def shuffle_data(self):
        self.buffer.shuffle_data()
