"""Courtesy of https://github.com/openai/random-network-distillation"""
from copy import copy

import numpy as np

import gym

from algorithms.env_wrappers import unwrap_env


class StickyActionWrapper(gym.Wrapper):
    def __init__(self, env, p=0.25):
        super(StickyActionWrapper, self).__init__(env)
        self.p = p
        self.last_action = 0

    def reset(self):
        self.last_action = 0
        return self.env.reset()

    def step(self, action):
        if self.unwrapped.np_random.uniform() < self.p:
            action = self.last_action
        self.last_action = action
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info


class MaxAndSkipWrapper(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_to_maxpool = 2
        self._obs_buffer = np.zeros((self._obs_to_maxpool,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)

            if i >= self._skip - self._obs_to_maxpool:
                self._obs_buffer[self._skip - i - 1] = obs

            total_reward += reward
            if done:
                break

        # Note that the observation on the done=True frame doesn't matter
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info


class AtariVisitedRoomsInfoWrapper(gym.Wrapper):
    """This is good for summaries, to monitor training progress."""

    def __init__(self, env):
        super(AtariVisitedRoomsInfoWrapper, self).__init__(env)
        env_id = env.unwrapped.spec.id
        self.room_address = 3 if 'Montezuma' in env_id else 1
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap_env(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())
        if done:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode'].update(visited_rooms=copy(self.visited_rooms))
            self.visited_rooms.clear()
        return obs, rew, done, info

    def reset(self):
        return self.env.reset()
