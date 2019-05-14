"""Courtesy of OpenAI Baselines, distributions.py"""
import math

import numpy as np
import tensorflow as tf

_EPS = 1e-9  # to prevent numerical problems such as ln(0)


class DiagGaussianPd:
    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / (self.std + _EPS)), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi + _EPS) * tf.to_float(tf.shape(x)[-1]) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        assert isinstance(other, DiagGaussianPd)
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) / (
                2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + 0.5 * np.log(2.0 * np.pi * np.e + _EPS), axis=-1)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))


class CategoricalProbabilityDistribution:
    def __init__(self, logits):
        self.logits = logits

    @property
    def num_categories(self):
        return self.logits.get_shape().as_list()[-1]

    def probabilities(self):
        """Vector of probabilities for all outcomes."""
        return tf.nn.softmax(self.logits)

    def probability(self, outcome):
        """Probability of the particular outcome under the distribution."""
        outcome_one_hot = tf.one_hot(outcome, depth=self.num_categories)
        probabilities = self.probabilities()
        return tf.reduce_sum(outcome_one_hot * probabilities, axis=-1)

    def entropy(self):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.log(z0 + _EPS) - a0), axis=-1)

    def max_entropy(self):
        """Maximum possible entropy for this probability distribution."""
        avg_prob = 1.0 / self.num_categories
        return -math.log(avg_prob)

    def kl(self, other_logits):
        a0 = self.logits - tf.reduce_max(self.logits, axis=-1, keepdims=True)
        a1 = other_logits - tf.reduce_max(other_logits, axis=-1, keepdims=True)
        ea0 = tf.exp(a0)
        ea1 = tf.exp(a1)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        z1 = tf.reduce_sum(ea1, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (a0 - tf.log(z0 + _EPS) - a1 + tf.log(z1 + _EPS)), axis=-1)

    def sample(self):
        u = tf.random_uniform(tf.shape(self.logits))
        return tf.argmax(self.logits - tf.log(-tf.log(u + _EPS)), axis=-1)
