import gym
# noinspection PyUnresolvedReferences
import vizdoomgym

from algorithms.env_wrappers import ResizeAndGrayscaleWrapper, StackFramesWrapper, RewardScalingWrapper, \
    SkipAndStackFramesWrapper, TimeLimitWrapper, RemainingTimeWrapper
from utils.envs.doom.wrappers.observation_space import SetResolutionWrapper
from utils.envs.doom.wrappers.step_human_input import StepHumanInput

DOOM_W = DOOM_H = 84


class DoomCfg:
    def __init__(self, name, env_id, reward_scaling, default_timeout):
        self.name = name
        self.env_id = env_id
        self.reward_scaling = reward_scaling
        self.default_timeout = default_timeout
        self.has_timer = False


DOOM_ENVS = [
    DoomCfg('basic', 'VizdoomBasic-v0', 0.01, 300),
    DoomCfg('maze', 'VizdoomMyWayHome-v0', 1.0, 2100),
    DoomCfg('maze_sparse', 'VizdoomMyWayHomeSparse-v0', 1.0, 2100),
    DoomCfg('maze_very_sparse', 'VizdoomMyWayHomeVerySparse-v0', 1.0, 2100),
]


def env_by_name(name):
    for cfg in DOOM_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown Doom env')


def make_doom_env(doom_cfg, mode='train', has_timer=False, human_input=False):
    env = gym.make(doom_cfg.env_id)

    if human_input:
        env = StepHumanInput(env)

    # courtesy of https://github.com/pathak22/noreward-rl/blob/master/src/envs.py
    # and https://github.com/ppaquette/gym-doom
    if mode == 'test':
        env = SetResolutionWrapper(env, '800x600')
    else:
        env = SetResolutionWrapper(env, '160x120')

    env = ResizeAndGrayscaleWrapper(env, DOOM_W, DOOM_H)

    timeout = doom_cfg.default_timeout - 10
    env = TimeLimitWrapper(env, limit=timeout, random_variation_steps=5)

    if mode == 'test':
        # disable action repeat during test time
        env = StackFramesWrapper(env, stack_past_frames=4)
    else:
        # during training we repeat the last action n times and stack the same number of frames to capture dynamics
        env = SkipAndStackFramesWrapper(env, num_frames=4)

    env = RewardScalingWrapper(env, doom_cfg.reward_scaling)

    if has_timer:
        env = RemainingTimeWrapper(env)

    return env
