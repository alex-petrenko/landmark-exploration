import math

import tensorflow as tf

from unittest import TestCase

from utils.distributions import CategoricalProbabilityDistribution, _EPS


class TestDistr(TestCase):
    def test_categorical(self):
        g = tf.Graph()
        with g.as_default():
            logits_uniform = tf.constant([0, 0, 0, 0], dtype=tf.float32)
            uniform = CategoricalProbabilityDistribution(logits_uniform)

            n = uniform.num_categories
            self.assertEqual(n, 4)

            uniform_entropy = uniform.entropy()
            uniform_probs = uniform.probabilities()
            kl_self = uniform.kl(logits_uniform)

        with tf.Session(graph=g) as sess:
            e, probs, kl_self_v = sess.run([uniform_entropy, uniform_probs, kl_self])
            self.assertAlmostEqual(e, -math.log(0.25))
            for p in probs:
                self.assertAlmostEqual(p, 0.25)
            self.assertAlmostEqual(kl_self_v, 0)

            for i in range(4):
                p = sess.run(uniform.probability(i))
                self.assertAlmostEqual(p, 0.25)

            self.assertAlmostEqual(e, uniform.max_entropy(), places=5)

        with g.as_default():
            logits = tf.constant([0, 1, 2, 3], dtype=tf.float32)
            distr = CategoricalProbabilityDistribution(logits)
            entropy = distr.entropy()
            probs = distr.probabilities()
            kl_self = distr.kl(logits)
            kl_uniform = distr.kl(logits_uniform)

        with tf.Session(graph=g) as sess:
            e, probs, kl_self_v, kl_uniform_v = sess.run([entropy, probs, kl_self, kl_uniform])
            self.assertAlmostEqual(kl_self_v, 0)
            self.assertGreater(kl_uniform_v, _EPS)

            self.assertAlmostEqual(uniform.max_entropy(), distr.max_entropy())

        g.finalize()
