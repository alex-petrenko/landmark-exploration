import os
import shutil
import time
from os.path import join

import cv2
import deepmind_lab
import gym
import numpy as np
from gym.utils import seeding

from utils.utils import project_root, ensure_dir_exists

ACTION_SET = (
    (0, 0, 0, 0, 0, 0, 0),  # Idle
    (0, 0, 0, 1, 0, 0, 0),  # Forward
    (0, 0, 0, -1, 0, 0, 0),  # Backward
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),  # Look Right
)


class LevelCache:
    def __init__(self, cache_dir):
        ensure_dir_exists(cache_dir)
        self._cache_dir = cache_dir

    def fetch(self, key, pk3_path):
        path = join(self._cache_dir, key)

        if os.path.isfile(path):
            # copy the cached file to the path expected by DeepMind Lab
            shutil.copyfile(path, pk3_path)
            return True

        return False

    def write(self, key, pk3_path):
        path = os.path.join(self._cache_dir, key)

        if not os.path.isfile(path):
            # copy the cached file DeepMind Lab has written to the cache directory
            shutil.copyfile(pk3_path, path)


level_cache = LevelCache(join(project_root(), '.dmlab_cache'))


class DmlabGymEnv(gym.Env):
    def __init__(self, level, action_repeat, extra_cfg=None):
        self._width = 96
        self._height = 72
        self._main_observation = 'BGR_INTERLEAVED'
        self._action_repeat = action_repeat

        self._random_state = None

        observation_format = [self._main_observation, 'DEBUG.POS.TRANS']
        config = {'width': self._width, 'height': self._height}
        if extra_cfg is not None:
            config.update(extra_cfg)
        config = {k: str(v) for k, v in config.items()}

        renderer = 'hardware'

        self._dmlab = deepmind_lab.Lab(
            level, observation_format, config=config, renderer=renderer, level_cache=level_cache,
        )

        self._action_set = ACTION_SET
        self._action_list = np.array(self._action_set, dtype=np.intc)  # DMLAB requires intc type for actions

        self._last_observation = None

        self._render_scale = 5
        self._render_fps = 30
        self._last_frame = time.time()

        self.action_space = gym.spaces.Discrete(len(self._action_set))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)

        self.seed()

    def seed(self, seed=None):
        initial_seed = seeding.hash_seed(seed) % 2 ** 32
        self._random_state = np.random.RandomState(seed=initial_seed)
        return [initial_seed]

    def reset(self):
        self._dmlab.reset(seed=self._random_state.randint(0, 2 ** 31 - 1))
        self._last_observation = self._dmlab.observations()[self._main_observation]
        return self._last_observation

    def step(self, action):
        reward = self._dmlab.step(self._action_list[action], num_steps=self._action_repeat)
        done = not self._dmlab.is_running()
        if not done:
            self._last_observation = self._dmlab.observations()[self._main_observation]

        info = {'num_frames': self._action_repeat}
        return self._last_observation, reward, done, info

    def render(self, mode='human'):
        if self._last_observation is None and self._dmlab.is_running():
            self._last_observation = self._dmlab.observations()[self._main_observation]

        img_rgb = self._last_observation
        if mode == 'rgb_array':
            return img_rgb
        elif mode != 'human':
            raise Exception(f'Rendering mode {mode} not supported')

        scale = self._render_scale
        img_big = cv2.resize(img_rgb, (self._width * scale, self._height * scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('atari', img_big)

        since_last_frame = time.time() - self._last_frame
        wait_time_sec = max(1.0 / self._render_fps - since_last_frame, 0.001)
        wait_time_ms = max(int(1000 * wait_time_sec), 1)
        cv2.waitKey(wait_time_ms)
        self._last_frame = time.time()

    def close(self):
        self._dmlab.close()
