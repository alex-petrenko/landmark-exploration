import gc
import shutil

import numpy as np
import tensorflow as tf

from unittest import TestCase

from algorithms.agent import AgentLearner, AgentRandom
from algorithms.algo_utils import RunningMeanStd, extract_keys
from algorithms.buffer import Buffer
from algorithms.encoders import is_normalized, tf_normalize
from algorithms.env_wrappers import TimeLimitWrapper, main_observation_space
from algorithms.exploit import run_policy_loop
from algorithms.tests.test_wrappers import TEST_ENV_NAME
from algorithms.tf_utils import placeholder_from_space
from utils.envs.doom.doom_utils import make_doom_env, doom_env_by_name
from utils.timing import Timing
from utils.utils import log


class TestAlgos(TestCase):
    def test_summary_step(self):
        params = AgentLearner.AgentParams('__test__')
        agent = AgentLearner(params)

        self.assertFalse(agent._should_write_summaries(0))
        self.assertTrue(agent._should_write_summaries(100 - 1))
        self.assertTrue(agent._should_write_summaries(200 - 1))

        self.assertTrue(agent._should_write_summaries(1002000 - 1))
        self.assertFalse(agent._should_write_summaries(1001000 - 1))
        self.assertFalse(agent._should_write_summaries(1000100 - 1))

        shutil.rmtree(params.experiment_dir())

    def test_run_loop(self):
        env = TimeLimitWrapper(make_doom_env(doom_env_by_name(TEST_ENV_NAME), mode='test'), 50, 0)

        def make_env_func():
            return env

        agent = AgentRandom(make_env_func, {}, close_env=False)
        run_policy_loop(agent, env, 1, 200)


class TestAlgoUtils(TestCase):
    # noinspection PyTypeChecker
    def test_running_mean_std(self):
        running_mean_std = RunningMeanStd(max_past_samples=100000)

        true_mu, true_sigma, batch_size = -1, 3, 256

        x = np.random.normal(true_mu, true_sigma, batch_size)

        running_mean_std.update(x)

        # after 1 batch we should have almost the exact same
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        self.assertAlmostEqual(running_mean_std.mean, batch_mean, places=5)
        self.assertAlmostEqual(running_mean_std.var, batch_var, places=5)
        self.assertAlmostEqual(running_mean_std.count, batch_size, places=3)

        # after many batches we should have an accurate estimate
        for _ in range(1000):
            x = np.random.normal(true_mu, true_sigma, batch_size)
            running_mean_std.update(x)

        log.info('estimated mean %.2f variance %.2f', running_mean_std.mean, running_mean_std.var)
        self.assertAlmostEqual(running_mean_std.mean, true_mu, places=0)
        self.assertAlmostEqual(running_mean_std.var, true_sigma ** 2, places=0)

    def test_extract_keys(self):
        test_obs = [{'obs1': 1, 'obs2': 2}, {'obs1': 3, 'obs2': 4}]
        obs1, obs2 = extract_keys(test_obs, 'obs1', 'obs2')
        self.assertEqual(obs1, [1, 3])
        self.assertEqual(obs2, [2, 4])


class TestEncoders(TestCase):
    def test_normalize(self):
        env = make_doom_env(doom_env_by_name(TEST_ENV_NAME))
        obs_space = main_observation_space(env)

        env.reset()
        obs = [env.step(0)[0] for _ in range(10)]

        self.assertTrue(np.all(obs_space.low == 0))
        self.assertTrue(np.all(obs_space.high == 255))
        self.assertEqual(obs_space.dtype, np.uint8)

        self.assertFalse(is_normalized(obs_space))

        tf.reset_default_graph()

        ph_obs = placeholder_from_space(obs_space)
        obs_tensor = tf_normalize(ph_obs, obs_space)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            normalized_obs = sess.run(obs_tensor, feed_dict={ph_obs: obs})

            self.assertEqual(normalized_obs.dtype, np.float32)
            self.assertLessEqual(normalized_obs.max(), 1.0)
            self.assertGreaterEqual(normalized_obs.min(), -1.0)

        tf.reset_default_graph()
        gc.collect()


class TestBuffer(TestCase):
    def test_buffer(self):
        buff = Buffer()

        buff.add(a=1, b='b', c=None, d=3.14)
        self.assertEqual(len(buff), 1)
        self.assertGreaterEqual(buff._capacity, 1)

        self.assertEqual(buff.a[0], 1)
        self.assertEqual(buff.b[0], 'b')

        buff.add_many(a=[2, 3], b=['c', 'd'], c=[None, list()], d=[2.71, 1.62])
        self.assertEqual(len(buff), 3)
        self.assertGreaterEqual(buff._capacity, 3)

        self.assertTrue(np.array_equal(buff.a, [1, 2, 3]))
        self.assertTrue(np.array_equal(buff.b, ['b', 'c', 'd']))

        buff.trim_at(5)
        self.assertTrue(np.array_equal(buff.a, [1, 2, 3]))

        buff.trim_at(2)
        self.assertTrue(np.array_equal(buff.a, [1, 2]))

        buff.add_many(a=[2, 3], b=['c', 'd'], c=[None, list()], d=[2.71, 1.62])

        buff.shuffle_data()
        buff.shuffle_data()
        buff.shuffle_data()

        buff.trim_at(1)
        self.assertIn(buff.a[0], [1, 2, 3])

        self.assertEqual(len(buff), 1)
        self.assertGreaterEqual(buff._capacity, 4)

        buff_temp = Buffer()
        buff_temp.add(a=10, b='e', c=dict(), d=9.81)

        buff.add_buff(buff_temp)

        self.assertEqual(len(buff), 2)

        buff.clear()
        self.assertEqual(len(buff), 0)

    def test_buffer_performance(self):
        small_buffer = Buffer()
        small_buffer.add_many(obs=np.zeros([1000, 42, 42, 1]))

        buffer = Buffer()

        t = Timing()

        with t.timeit('add'):
            for i in range(100):
                buffer.add_buff(small_buffer)

        huge_buffer = Buffer()
        with t.timeit('add_huge'):
            huge_buffer.add_buff(buffer)
            huge_buffer.add_buff(buffer)

        with t.timeit('clear_and_add'):
            huge_buffer.clear()
            huge_buffer.add_buff(buffer)
            huge_buffer.add_buff(buffer)

        with t.timeit('shuffle'):
            huge_buffer.shuffle_data()

        log.debug('Timing: %s', t)

    def test_buffer_shuffle(self):
        b = Buffer()
        b.add_many(a=np.arange(10000), b=np.arange(10000))

        for i in range(5):
            self.assertTrue(np.array_equal(b.a, b.b))
            b.shuffle_data()
