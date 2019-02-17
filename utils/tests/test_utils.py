import time
from unittest import TestCase

import numpy as np

from algorithms.algo_utils import EPS
from utils.decay import LinearDecay
from utils.timing import TimingContext, Timing
from utils.utils import numpy_all_the_way, numpy_flatten, max_with_idx, min_with_idx, AttrDict


class TestDecay(TestCase):
    def test_linear_decay(self):
        with self.assertRaises(Exception):
            LinearDecay([])

        def chk(value, expected):
            self.assertAlmostEqual(value, expected)

        decay = LinearDecay([(0, 1)])
        chk(decay.at(0), 1)
        chk(decay.at(100), 1)

        decay = LinearDecay([(0, 0), (1000, 1)])
        chk(decay.at(-1), 0)
        chk(decay.at(0), 0)
        chk(decay.at(1000), 1)
        chk(decay.at(10000), 1)
        chk(decay.at(500), 0.5)
        chk(decay.at(450), 0.45)

        decay = LinearDecay([(0, 0), (1000, 1), (2000, 5)], staircase=0.1)
        chk(decay.at(-1), 0)
        chk(decay.at(0), 0)
        chk(decay.at(1000), 1)
        chk(decay.at(1500), 3)
        chk(decay.at(1501), 3)
        chk(decay.at(2000), 5)
        chk(decay.at(10000), 5)
        chk(decay.at(500), 0.5)
        chk(decay.at(401), 0.4)
        chk(decay.at(450), 0.4)
        chk(decay.at(499), 0.4)

        decay = LinearDecay([(0, 50), (100, 100)], staircase=100)
        chk(decay.at(0), 50)
        chk(decay.at(1), 50)
        chk(decay.at(99), 50)


class TestNumpyUtil(TestCase):
    def test_numpy_all_the_way(self):
        a = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        lst = [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])]
        numpy_arr = numpy_all_the_way(lst)
        self.assertTrue(np.array_equal(a, numpy_arr))

    def test_numpy_flatten(self):
        a = np.arange(9)
        lst = [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8])]
        flattened = numpy_flatten(lst)
        self.assertTrue(np.array_equal(a, flattened))


class TestTiming(TestCase):
    def test_simple_timer(self):
        timing = Timing()
        with timing.timeit('test'):
            time.sleep(0.05)
        self.assertGreater(timing.test, 0.05 - EPS)


class TestUtil(TestCase):
    def test_op_with_idx(self):
        x = [1, 3, 2, 10, -1, 5, 9]
        max_x, max_idx = max_with_idx(x)
        self.assertEqual(max_x, max(x))
        self.assertEqual(max_idx, 3)

        min_x, min_idx = min_with_idx(x)
        self.assertEqual(min_x, min(x))
        self.assertEqual(min_idx, 4)
