import time

import cv2
import numpy as np

import gym
import deepmind_lab
from gym.utils import seeding


ACTION_SET = (
    (0, 0, 0, 0, 0, 0, 0),    # Idle
    (0, 0, 0, 1, 0, 0, 0),    # Forward
    (0, 0, 0, -1, 0, 0, 0),   # Backward
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),   # Look Right
)


class DmlabGymEnv(gym.Env):
    def __init__(self):
        self._width = 96
        self._height = 72
        self._main_observation = 'RGB_INTERLEAVED'
        self._action_repeat = 4

        self._random_state = None

        level_name = 'contributed/dmlab30/explore_goal_locations_large'
        observation_format = [self._main_observation, 'DEBUG.POS.TRANS']
        config = {'width': str(self._width), 'height': str(self._height), 'fps': str(30)}
        renderer = 'hardware'

        self._dmlab = deepmind_lab.Lab(level_name, observation_format, config=config, renderer=renderer)
        self._action_set = ACTION_SET
        self._action_list = np.array(self._action_set, dtype=np.intc)  # DMLAB requires intc type for actions

        self._last_observation = None

        self._render_scale = 5
        self._render_fps = 15
        self._last_frame = time.time()

        self.action_space = gym.spaces.Discrete(len(self._action_set))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8)

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
        return self._last_observation, reward, done, {}

    def render(self, mode='human'):
        img_rgb = self._dmlab.observations()[self._main_observation]
        if mode == 'rgb_array':
            return img_rgb
        elif mode != 'human':
            raise Exception(f'Rendering mode {mode} not supported')

        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        scale = self._render_scale
        img_big = cv2.resize(img_rgb, (self._width * scale, self._height * scale), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('atari', img_big)

        since_last_frame = time.time() - self._last_frame
        wait_time_sec = max(1.0 / self._render_fps - since_last_frame, 0.001)
        wait_time_ms = max(int(1000 * wait_time_sec), 1)
        # cv2.waitKey(wait_time_ms)
        time.sleep(wait_time_sec)
        self._last_frame = time.time()

    def close(self):
        self._dmlab.close()
