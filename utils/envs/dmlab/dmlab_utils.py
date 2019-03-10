from algorithms.env_wrappers import ResizeWrapper
from utils.envs.dmlab.dmlab_gym import DmlabGymEnv


class DmLabCfg:
    def __init__(self, name, env_id, default_timeout):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout


DMLAB_ENVS = [
    DmLabCfg('dmlab_sparse', '', default_timeout=18000),
]


def dmlab_env_by_name(name):
    for cfg in DMLAB_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown DMLab env')


def make_dmlab_env(dmlab_cfg, mode='train'):
    env = DmlabGymEnv()
    env = ResizeWrapper(env, 84, 84, grayscale=False, add_channel_dim=False, area_interpolation=False)
    return env
