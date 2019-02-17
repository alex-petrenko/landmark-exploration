import time

from utils.utils import AttrDict


class TimingContext:
    def __init__(self, timer, key):
        self._timer = timer
        self._key = key

    def __enter__(self):
        self._timer[self._key] = time.time()

    def __exit__(self, type_, value, traceback):
        self._timer[self._key] = time.time() - self._timer[self._key]


class Timing(AttrDict):
    def __init__(self, d=None):
        super(Timing, self).__init__(d)

    def timeit(self, key):
        return TimingContext(self, key)
