import gym

from algorithms.env_wrappers import ResizeAndGrayscaleWrapper, ClipRewardWrapper
from utils.envs.atari.atari_wrappers import StickyActionWrapper, MaxAndSkipWrapper, AtariVisitedRoomsInfoWrapper


ATARI_W = ATARI_H = 84


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


def make_atari_env(atari_cfg):
    """Heavily inspired by https://github.com/openai/random-network-distillation"""

    env = gym.make(atari_cfg.env_id)
    env._max_episode_steps = atari_cfg.default_timeout

    assert 'NoFrameskip' in env.spec.id

    env = StickyActionWrapper(env)
    env = MaxAndSkipWrapper(env, skip=4)
    if 'Montezuma' in atari_cfg.env_id or 'Pitfall' in atari_cfg.env_id:
        env = AtariVisitedRoomsInfoWrapper(env)

    env = ResizeAndGrayscaleWrapper(env, ATARI_W, ATARI_H)
    env = ClipRewardWrapper(env)
    return env
