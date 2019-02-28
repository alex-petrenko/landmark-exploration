import time

from algorithms.algo_utils import EPS
from utils.utils import AttrDict


class TimingContext:
    def __init__(self, timer, key):
        self._timer = timer
        self._key = key

    def __enter__(self):
        self._timer[self._key] = time.time()

    def __exit__(self, type_, value, traceback):
        self._timer[self._key] = max(time.time() - self._timer[self._key], EPS)  # EPS to prevent div by zero


class Timing(AttrDict):
    def __init__(self, d=None):
        super(Timing, self).__init__(d)

    def timeit(self, key):
        return TimingContext(self, key)

    def __str__(self):
        s = ''
        for key, value in self.items():
            s += f'{key}: {value:.3f}, '
        return s
