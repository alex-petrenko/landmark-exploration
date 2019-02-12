"""Courtesy of https://github.com/openai/random-network-distillation"""
import time
from copy import copy

import cv2
import numpy as np

import gym

from algorithms.env_wrappers import unwrap_env
from utils.utils import log


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
        total_reward, num_frames = 0.0, 0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            num_frames += 1

            if i >= self._skip - self._obs_to_maxpool:
                self._obs_buffer[self._skip - i - 1] = obs

            total_reward += reward
            if done:
                break

        info['num_frames'] = num_frames

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

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())
        if done:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode'].update(visited_rooms=copy(self.visited_rooms))
            self.visited_rooms.clear()
        return obs, rew, done, info


class RenderWrapper(gym.Wrapper):
    def __init__(self, env, render_w=420, render_h=420, fps=25):
        super(RenderWrapper, self).__init__(env)
        self.w = render_w
        self.h = render_h
        self.last_frame = time.time()
        self.fps = fps

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self, mode='human', **kwargs):
        atari_img = self.env.render(mode='rgb_array')

        if mode == 'rgb_array':
            return atari_img
        elif mode != 'human':
            raise Exception(f'Rendering mode {mode} not supported')

        atari_img = cv2.cvtColor(atari_img, cv2.COLOR_BGR2RGB)
        atari_img_big = cv2.resize(atari_img, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('atari', atari_img_big)

        since_last_frame = time.time() - self.last_frame
        wait_time_sec = max(1.0 / self.fps - since_last_frame, 0.001)
        wait_time_ms = max(int(1000 * wait_time_sec), 1)
        cv2.waitKey(wait_time_ms)
        self.last_frame = time.time()
