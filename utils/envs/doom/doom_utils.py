import gym
# noinspection PyUnresolvedReferences
import vizdoomgym

from algorithms.env_wrappers import ResizeWrapper, RewardScalingWrapper, SkipFramesWrapper
from utils.envs.doom.wrappers.observation_space import SetResolutionWrapper
from utils.envs.doom.wrappers.step_human_input import StepHumanInput

DOOM_W = DOOM_H = 84


class DoomCfg:
    def __init__(self, name, env_id, reward_scaling, default_timeout):
        self.name = name
        self.env_id = env_id
        self.reward_scaling = reward_scaling
        self.default_timeout = default_timeout


DOOM_ENVS = [
    DoomCfg('doom_basic', 'VizdoomBasic-v0', 0.01, 300),
    DoomCfg('doom_maze', 'VizdoomMyWayHome-v0', 1.0, 2100),
    DoomCfg('doom_maze_sparse', 'VizdoomMyWayHomeSparse-v0', 1.0, 2100),
    DoomCfg('doom_maze_very_sparse', 'VizdoomMyWayHomeVerySparse-v0', 1.0, 2100),

    DoomCfg('doom_maze_goal', 'VizdoomMyWayHomeGoal-v0', 1.0, 2100),
]


def doom_env_by_name(name):
    for cfg in DOOM_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown Doom env')


def make_doom_env(doom_cfg, mode='train', human_input=False):
    env = gym.make(doom_cfg.env_id)

    if human_input:
        env = StepHumanInput(env)

    # courtesy of https://github.com/pathak22/noreward-rl/blob/master/src/envs.py
    # and https://github.com/ppaquette/gym-doom
    if mode == 'test':
        env = SetResolutionWrapper(env, '800x600')
    else:
        env = SetResolutionWrapper(env, '160x120')

    env = ResizeWrapper(env, DOOM_W, DOOM_H, grayscale=False)

    if mode != 'test':
        env = SkipFramesWrapper(env, skip_frames=4)

    if doom_cfg.reward_scaling != 1.0:
        env = RewardScalingWrapper(env, doom_cfg.reward_scaling)

    return env
