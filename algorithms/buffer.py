import numpy as np


class Buffer:
    """Generic experience buffer class."""
    def __init__(self):
        self._data = {}

        # assuming all buffers have the exact same size
        self._size = self._capacity = 0

    def _ensure_enough_space(self, space_required):
        assert len(self._data) >= 1  # we need to have some elements already, to determine required memory
        assert self._size <= self._capacity

        if self._capacity >= self._size + space_required:
            # we already have enough space
            return

        capacity_delta = max(self._capacity // 2, 10)  # ensure exponentially low number of reallocs
        capacity_delta = max(capacity_delta, self._size + space_required - self._capacity)
        for key in self._data.keys():
            self._data[key].resize((self._capacity + capacity_delta, ) + self._data[key].shape[1:], refcheck=False)
        self._capacity += capacity_delta

        assert self._capacity >= self._size + space_required

    def add(self, **kwargs):
        """Append one-step data to the experience buffer."""
        new_size = self._size

        for key, value in kwargs.items():
            if key not in self._data:
                np_arr = np.asarray([value])
                self._data[key] = np.empty_like(np_arr)
                new_size = self._capacity = 1
            else:
                self._ensure_enough_space(1)
                new_size = self._size + 1

            self._data[key][self._size] = value

        self._size = new_size
        assert self._size <= self._capacity

    def add_many(self, max_to_add=1000000000, **kwargs):
        new_size = self._size

        for key, value in kwargs.items():
            size = min(len(value), max_to_add)
            if size <= 0:
                continue

            if key not in self._data:
                np_arr = np.asarray(value[:size])
                self._data[key] = np.empty_like(np_arr)
                new_size = self._capacity = size
            else:
                self._ensure_enough_space(size)
                new_size = self._size + size

            self._data[key][self._size:self._size + size] = value[:size]

        self._size = new_size
        assert self._size <= self._capacity

    # noinspection PyProtectedMember
    def add_buff(self, buff, max_to_add=1000000000):
        if len(buff) <= 0:
            return

        kwargs = {key: getattr(buff, key) for key in buff._data.keys()}
        self.add_many(max_to_add, **kwargs)

    def shuffle_data(self):
        if self._size <= 0:
            return

        rng_state = np.random.get_state()
        for key in self._data.keys():
            np.random.set_state(rng_state)
            np.random.shuffle(self._data[key][:self._size])

    def trim_at(self, new_size):
        """Discard some data from the end of the buffer, but keep the capacity."""
        if new_size >= self._size:
            return
        self._size = new_size

    def clear(self):
        self.trim_at(0)

    def empty(self):
        return len(self) == 0

    def __len__(self):
        return self._size

    def __getattr__(self, key):
        return self._data[key][:self._size]
