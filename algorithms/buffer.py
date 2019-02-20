import numpy as np


class Buffer:
    """Generic experience buffer class."""
    def __init__(self):
        self._data = {}

        # assuming all buffers have the exact same size
        self._size = self._capacity = 0

    def _ensure_enough_space(self, space_required):
        assert self._size >= 1  # we need to have at least one element already to be able to allocate space
        assert self._size <= self._capacity

        if self._capacity >= self._size + space_required:
            # we already have enough space
            return

        capacity_delta = max(0.5 * self._capacity, 10)  # ensure exponentially low number of reallocs
        capacity_delta = max(capacity_delta, space_required)
        for key in self._data.keys():
            self._data[key].resize((self._capacity + capacity_delta, ) + self._data[key].shape[1:])
        self._capacity += capacity_delta

        assert self._capacity >= self._size + space_required

    def add(self, **kwargs):
        """Append one-step data to the experience buffer."""
        new_size = self._size

        for key, value in kwargs.items():
            if key not in self._data:
                self._data[key] = np.asarray([value])
                new_size = self._capacity = 1
            else:
                self._ensure_enough_space(1)
                self._data[key][self._size] = value
                new_size = self._size + 1

        self._size = new_size
        assert self._size <= self._capacity

    def add_many(self, **kwargs):
        new_size = self._size

        for key, value in kwargs.items():
            size = len(value)

            if key not in self._data:
                self._data[key] = np.asarray(value)
                new_size = self._capacity = size
            else:
                self._ensure_enough_space(size)
                self._data[key][self._size:self._size + size] = value
                new_size = self._size + size

        self._size = new_size
        assert self._size <= self._capacity

    def add_buff(self, buff):
        kwargs = {key: getattr(buff, key) for key in self._data.keys()}
        self.add_many(**kwargs)

    def shuffle_data(self):
        if self._size <= 0:
            return

        chaos = np.random.permutation(self._size)
        for key in self._data.keys():
            self._data[key][:self._size] = self._data[key][chaos]

    def trim_at(self, new_size):
        """Discard some data from the end of the buffer, but keep the capacity."""
        if new_size >= self._size:
            return
        self._size = new_size

    def clear(self):
        self.trim_at(0)

    def __len__(self):
        return self._size

    def __getattr__(self, key):
        return self._data[key][:self._size]
