import gym
from gym.envs.atari.atari_env import ACTION_MEANING
from pynput.keyboard import Key

from algorithms.env_wrappers import ResizeWrapper, ClipRewardWrapper
from utils.envs.atari.atari_wrappers import StickyActionWrapper, MaxAndSkipWrapper, AtariVisitedRoomsInfoWrapper, \
    RenderWrapper, OneLifeWrapper
from utils.utils import log

ATARI_W = ATARI_H = 84


def action_name_to_action(action_name):
    for action, name in ACTION_MEANING.items():
        if name == action_name:
            return action

    log.warning('Unknown action %s', action_name)
    return None


action_table = {
    Key.space: 'FIRE',
    Key.up: 'UP',
    Key.down: 'DOWN',
    Key.left: 'LEFT',
    Key.right: 'RIGHT',
}


def key_to_action(key):
    if key not in action_table:
        return None

    action_name = action_table[key]
    return action_name_to_action(action_name)


class AtariCfg:
    def __init__(self, name, env_id, default_timeout):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout
        self.has_timer = False


ATARI_ENVS = [
    AtariCfg('atari_montezuma', 'MontezumaRevengeNoFrameskip-v4', default_timeout=18000),
]


def atari_env_by_name(name):
    for cfg in ATARI_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown Atari env')


def make_atari_env(atari_cfg, mode='train'):
    """Heavily inspired by https://github.com/openai/random-network-distillation"""

    env = gym.make(atari_cfg.env_id)
    env._max_episode_steps = atari_cfg.default_timeout

    assert 'NoFrameskip' in env.spec.id

    env = OneLifeWrapper(env)
    env = StickyActionWrapper(env)
    env = MaxAndSkipWrapper(env, skip=4)

    if 'Montezuma' in atari_cfg.env_id or 'Pitfall' in atari_cfg.env_id:
        env = AtariVisitedRoomsInfoWrapper(env)

    env = ResizeWrapper(env, ATARI_W, ATARI_H, add_channel_dim=True, area_interpolation=True)
    env = ClipRewardWrapper(env)

    if mode == 'test':
        env = RenderWrapper(env)

    return env
