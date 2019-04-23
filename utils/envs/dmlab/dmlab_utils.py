from algorithms.utils.env_wrappers import TimeLimitWrapper
from utils.envs.dmlab.dmlab_gym import DmlabGymEnv


class DmLabCfg:
    def __init__(self, name, level, extra_cfg=None):
        self.name = name
        self.level = level
        self.default_timer = 1800
        self.extra_cfg = {} if extra_cfg is None else extra_cfg


DMLAB_ENVS = [
    DmLabCfg('dmlab_sparse', 'contributed/dmlab30/explore_goal_locations_large'),
    DmLabCfg(
        'dmlab_very_sparse', 'contributed/dmlab30/explore_goal_locations_large', extra_cfg={'minGoalDistance': '10'},
    ),
    DmLabCfg('dmlab_sparse_doors', 'contributed/dmlab30/explore_obstructed_goals_large'),
]


def dmlab_env_by_name(name):
    for cfg in DMLAB_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown DMLab env')


def make_dmlab_env(cfg, mode='train'):
    repeat_actions = 4
    if mode == 'test':
        repeat_actions = 1

    env = DmlabGymEnv(cfg.level, repeat_actions, cfg.extra_cfg)
    env = TimeLimitWrapper(env, 900, 0)
    return env

