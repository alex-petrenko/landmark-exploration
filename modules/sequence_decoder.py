import tensorflow as tf

from modules.distributions import DiagGaussianPd
from ops import autoregressive_lstm, batch_apply, fc


class Sequence_Decoder(object):
    """ An architecture that is able to decode an input feature vector into a sequence of output latents"""
    def __init__(self, name, is_train, config):
        self.name = name
        self._is_train = is_train
        self._reuse = False
        self.config = config

    def __call__(self, initial_state, rollout_len):
        reuse = self._reuse
        with tf.variable_scope(self.name, reuse=reuse):
            if not reuse:
                print('\033[93m'+self.name+'\033[0m')

            outputs = autoregressive_lstm(initial_state, rollout_len, self.config.lstm_latent_size,
                                          self.config.batch_size, info=not reuse, name='decoder_lstm')

            # map LSTM outputs to desired output latent size
            outputs = batch_apply(fc, outputs,
                                  output_shape=self.config.enc_latent_size,
                                  activation_fn=None, is_train=self._is_train, info=not reuse)

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return outputs


class Stochastic_Sequence_Decoder(object):
    """
    An architecture that is able to decode an input feature vector into a sequence of output latents
    while sampling from a prior distribution.
    """

    def __init__(self, name, is_train, config):
        self.name = name
        self._is_train = is_train
        self._reuse = False
        self.config = config

        self.tiled_last_past_input = None
        self.z_dist_posterior = self.z_dist_prior = None

    def __call__(self, initial_state, last_past_input, groundtruth_future_encodings, num_prior_decodings):
        reuse = self._reuse
        is_train = self._is_train
        with tf.variable_scope(self.name, reuse=reuse):
            if not reuse:
                print('\033[93m' + self.name + '\033[0m')

            # preprocess inputs
            rollout_len = groundtruth_future_encodings.get_shape().as_list()[1]
            reps = [1, rollout_len] + [1] * (len(last_past_input.get_shape().as_list()) - 1)
            self.tiled_last_past_input = tf.tile(tf.expand_dims(last_past_input, axis=1), reps)

            def inference_network(input_, name='Inference_Network'):
                with tf.variable_scope(name, reuse=reuse):
                    if not reuse:
                        print('\033[93m' + name + '\033[0m')
                    _ = input_
                    for i in range(self.config.inference_num_layers - 1):
                        _ = fc(_, 2*self.config.z_latent_size * (self.config.inference_num_layers - i),
                               is_train, info=not reuse, name='fc{}'.format(i + 1))
                    output = fc(_, 2*self.config.z_latent_size,
                                is_train, activation_fn=None, info=not reuse, name='fc{}'.format(i + 2))
                return output

            # inference/posterior network
            inf_input = tf.concat((self.tiled_last_past_input, groundtruth_future_encodings), axis=-1)
            z_dists_params_posterior = batch_apply(inference_network, inf_input)

            # prior network
            prior_lstm_output = autoregressive_lstm(initial_state, rollout_len, self.config.lstm_latent_size,
                                                    self.config.batch_size, info=not reuse, name='decoder_lstm')
            z_dists_params_prior = batch_apply(fc, prior_lstm_output,
                                               output_shape=2*self.config.z_latent_size, activation_fn=None,
                                               is_train=self._is_train, info=not reuse)

            # sample z from distribution
            with tf.name_scope('Sample_Distributions'):
                mean_posterior = z_dists_params_posterior[:, :, :self.config.z_latent_size]
                logstd_posterior = z_dists_params_posterior[:, :, self.config.z_latent_size:]
                mean_prior = z_dists_params_prior[:, :, :self.config.z_latent_size]
                logstd_prior = z_dists_params_prior[:, :, self.config.z_latent_size:]
                self.z_dist_posterior = DiagGaussianPd(mean_posterior, logstd_posterior)
                self.z_dist_prior = DiagGaussianPd(mean_prior, logstd_prior)
                posterior_sample = self.z_dist_posterior.sample()
                prior_sample = [self.z_dist_prior.sample() for _ in range(num_prior_decodings)]

            # compute output latents
            pred_network_input_posterior = tf.concat((self.tiled_last_past_input, posterior_sample), axis=-1)
            output_latents_posterior = batch_apply(self._predictive_network, pred_network_input_posterior, self._reuse)

            pred_network_input_prior = [tf.concat((self.tiled_last_past_input, ps), axis=-1) for ps in prior_sample]
            output_latents_prior = [batch_apply(self._predictive_network, pred_input, reuse=True)
                                    for pred_input in pred_network_input_prior]

            self._reuse = True
            self.var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            return output_latents_posterior, output_latents_prior, self.z_dist_posterior, self.z_dist_prior

    def _predictive_network(self, input_, reuse, name='Predictive_Network'):
        with tf.variable_scope(name, reuse=reuse):
            if not reuse:
                print('\033[93m' + name + '\033[0m')
            _ = input_
            for i in range(self.config.inference_num_layers - 1):
                _ = fc(_, self.config.enc_latent_size * (self.config.inference_num_layers - i),
                       self._is_train, info=not reuse, name='fc{}'.format(i + 1))
            output = fc(_, self.config.enc_latent_size,
                        self._is_train, activation_fn=None, info=not reuse, name='fc{}'.format(i + 2))
        return output

    def sample_n_latents(self, distribution, n_samples):
        """Used for plots & heatmaps."""
        if not self._reuse:
            raise Exception('Graph not constructed yet!')

        with tf.variable_scope(self.name, reuse=self._reuse):  # this is ugly, better replace with make_template?
            output_latents = []
            for _ in range(n_samples):
                pred_network_input = tf.concat((self.tiled_last_past_input, distribution.sample()), axis=-1)
                output_latents.append(batch_apply(self._predictive_network, pred_network_input, reuse=True))
        return output_latents
