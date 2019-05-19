"""
Gym env wrappers that make the environment suitable for the RL algorithms.

"""

from collections import deque

import cv2
import gym
import numpy as np
# noinspection PyProtectedMember
from gym import spaces, RewardWrapper, ObservationWrapper

from algorithms.utils.algo_utils import num_env_steps
from utils.utils import numpy_all_the_way, log


def reset_with_info(env):
    """Sometimes we want to get info with the very first frame."""
    obs = env.reset()
    info = {}
    if hasattr(env.unwrapped, 'get_info_all'):
        info = env.unwrapped.get_info_all()  # info for the new episode
    return obs, info


def unwrap_env(wrapped_env):
    return wrapped_env.unwrapped


def is_goal_based_env(env):
    dict_obs = isinstance(env.observation_space, spaces.Dict)
    if not dict_obs:
        return False

    for key in ['obs', 'goal']:
        if key not in env.observation_space.spaces:
            return False

    return True


def main_observation_space(env):
    if hasattr(env.observation_space, 'spaces'):
        return env.observation_space.spaces['obs']
    else:
        return env.observation_space


def has_image_observations(observation_space):
    """It's a heuristic."""
    return len(observation_space.shape) >= 2


def wrap_env(env, stack_past_frames):
    if not has_image_observations(main_observation_space(env)):
        # vector observations
        env = NormalizeWrapper(env)

    env = StackFramesWrapper(env, stack_past_frames)

    return env


class StackFramesWrapper(gym.core.Wrapper):
    """
    Gym env wrapper to stack multiple frames.
    Useful for training feed-forward agents on dynamic games.
    """

    def __init__(self, env, stack_past_frames):
        super(StackFramesWrapper, self).__init__(env)
        if len(env.observation_space.shape) not in [1, 2]:
            raise Exception('Stack frames works with vector observations and 2D single channel images')
        self._stack_past = stack_past_frames
        self._frames = None

        self._image_obs = has_image_observations(env.observation_space)

        if self._image_obs:
            new_obs_space_shape = env.observation_space.shape + (stack_past_frames,)
        else:
            new_obs_space_shape = list(env.observation_space.shape)
            new_obs_space_shape[0] *= stack_past_frames

        self.observation_space = spaces.Box(
            env.observation_space.low.flat[0],
            env.observation_space.high.flat[0],
            shape=new_obs_space_shape,
            dtype=env.observation_space.dtype,
        )

    def _render_stacked_frames(self):
        if self._image_obs:
            return np.transpose(numpy_all_the_way(self._frames), axes=[1, 2, 0])
        else:
            return np.array(self._frames).flatten()

    def reset(self):
        observation = self.env.reset()
        self._frames = deque([observation] * self._stack_past)
        return self._render_stacked_frames()

    def step(self, action):
        new_observation, reward, done, info = self.env.step(action)
        self._frames.popleft()
        self._frames.append(new_observation)
        return self._render_stacked_frames(), reward, done, info


class SkipFramesWrapper(gym.core.Wrapper):
    """Wrapper for action repeat over N frames to speed up training."""

    def __init__(self, env, skip_frames=4):
        super(SkipFramesWrapper, self).__init__(env)
        self._skip_frames = skip_frames

    def reset(self):
        return self.env.reset()

    def step(self, action):
        done = False
        info = None
        new_observation = None

        total_reward, num_frames = 0, 0
        for i in range(self._skip_frames):
            new_observation, reward, done, info = self.env.step(action)
            num_frames += 1
            total_reward += reward
            if done:
                break

        info['num_frames'] = num_frames
        return new_observation, total_reward, done, info


class SkipAndStackFramesWrapper(StackFramesWrapper):
    """Wrapper for action repeat + stack multiple frames to capture dynamics."""

    def __init__(self, env, skip_frames=4, stack_frames=3):
        super(SkipAndStackFramesWrapper, self).__init__(env, stack_past_frames=stack_frames)
        self._skip_frames = skip_frames

    def step(self, action):
        done = False
        info = {}
        total_reward, num_frames = 0, 0
        for i in range(self._skip_frames):
            new_observation, reward, done, info = self.env.step(action)
            num_frames += 1
            total_reward += reward
            self._frames.popleft()
            self._frames.append(new_observation)
            if done:
                break

        info['num_frames'] = num_frames
        return self._render_stacked_frames(), total_reward, done, info


class NormalizeWrapper(gym.core.Wrapper):
    """
    For environments with vector lowdim input.

    """

    def __init__(self, env):
        super(NormalizeWrapper, self).__init__(env)
        if len(env.observation_space.shape) != 1:
            raise Exception('NormalizeWrapper only works with lowdimensional envs')

        self.wrapped_env = env
        self._normalize_to = 1.0

        self._mean = (env.observation_space.high + env.observation_space.low) * 0.5
        self._max = env.observation_space.high

        self.observation_space = spaces.Box(
            -self._normalize_to, self._normalize_to, shape=env.observation_space.shape, dtype=np.float32,
        )

    def _normalize(self, obs):
        obs -= self._mean
        obs *= self._normalize_to / (self._max - self._mean)
        return obs

    def reset(self):
        observation = self.env.reset()
        return self._normalize(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return self._normalize(observation), reward, done, info

    @property
    def range(self):
        return [-self._normalize_to, self._normalize_to]


class ResizeWrapper(gym.core.Wrapper):
    """Resize observation frames to specified (w,h) and convert to grayscale."""

    def __init__(self, env, w, h, grayscale=True, add_channel_dim=False, area_interpolation=False):
        super(ResizeWrapper, self).__init__(env)

        self.w = w
        self.h = h
        self.grayscale = grayscale
        self.add_channel_dim = add_channel_dim
        self.interpolation = cv2.INTER_AREA if area_interpolation else cv2.INTER_NEAREST

        if isinstance(env.observation_space, spaces.Dict):
            new_spaces = {}
            for key, space in env.observation_space.spaces.items():
                new_spaces[key] = self._calc_new_obs_space(space)
            self.observation_space = spaces.Dict(new_spaces)
        else:
            self.observation_space = self._calc_new_obs_space(env.observation_space)

    def _calc_new_obs_space(self, old_space):
        low, high = old_space.low.flat[0], old_space.high.flat[0]

        if self.grayscale:
            new_shape = [self.w, self.h, 1] if self.add_channel_dim else [self.w, self.h]
        else:
            channels = old_space.shape[-1]
            new_shape = [self.w, self.h, channels]

        return spaces.Box(low, high, shape=new_shape, dtype=old_space.dtype)

    def _convert_obs(self, obs):
        obs = cv2.resize(obs, (self.w, self.h), interpolation=self.interpolation)
        if self.grayscale:
            obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)

        if self.add_channel_dim:
            return obs[:, :, None]  # add new dimension (expected by tensorflow)
        else:
            return obs

    def _observation(self, obs):
        if isinstance(obs, dict):
            new_obs = {}
            for key, value in obs.items():
                new_obs[key] = self._convert_obs(value)
            return new_obs
        else:
            return self._convert_obs(obs)

    def reset(self):
        return self._observation(self.env.reset())

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info


class RewardScalingWrapper(RewardWrapper):
    def __init__(self, env, scaling_factor):
        super(RewardScalingWrapper, self).__init__(env)
        self._scaling = scaling_factor
        self.reward_range = (r * scaling_factor for r in self.reward_range)

    def reward(self, reward):
        return reward * self._scaling


class TimeLimitWrapper(gym.core.Wrapper):
    terminated_by_timer = 'terminated_by_timer'

    def __init__(self, env, limit, random_variation_steps=0):
        super(TimeLimitWrapper, self).__init__(env)
        self._limit = limit
        self._variation_steps = random_variation_steps
        self._num_steps = 0
        self._terminate_in = self._random_limit()

    def _random_limit(self):
        return np.random.randint(-self._variation_steps, self._variation_steps + 1) + self._limit

    def reset(self):
        self._num_steps = 0
        self._terminate_in = self._random_limit()
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self._num_steps += num_env_steps([info])
        if done:
            pass
        else:
            if self._num_steps >= self._terminate_in:
                done = True
                info[self.terminated_by_timer] = True

        return observation, reward, done, info


class RemainingTimeWrapper(ObservationWrapper):
    """Designed to be used together with TimeLimitWrapper."""

    def __init__(self, env):
        super(RemainingTimeWrapper, self).__init__(env)

        # adding an additional input dimension to indicate time left before the end of episode
        self.observation_space = spaces.Dict({
            'timer': spaces.Box(0.0, 1.0, shape=[1], dtype=np.float32),
            'obs': env.observation_space,
        })

        wrapped_env = env
        while not isinstance(wrapped_env, TimeLimitWrapper):
            wrapped_env = wrapped_env.env
            if not isinstance(wrapped_env, gym.core.Wrapper):
                raise Exception('RemainingTimeWrapper is supposed to wrap TimeLimitWrapper')
        self.time_limit_wrapper = wrapped_env

    # noinspection PyProtectedMember
    def observation(self, observation):
        num_steps = self.time_limit_wrapper._num_steps
        terminate_in = self.time_limit_wrapper._terminate_in

        dict_obs = {
            'timer': num_steps / terminate_in,
            'obs': observation,
        }
        return dict_obs


class ClipRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        reward = min(1.0, reward)
        reward = max(-0.01, reward)
        return reward
