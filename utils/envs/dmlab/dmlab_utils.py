class DmLabCfg:
    def __init__(self, name, env_id, default_timeout):
        self.name = name
        self.env_id = env_id
        self.default_timeout = default_timeout


DMLAB_ENVS = [
    DmLabCfg('atari_montezuma', 'MontezumaRevengeNoFrameskip-v4', default_timeout=18000),
]


def dmlab_env_by_name(name):
    for cfg in DMLAB_ENVS:
        if cfg.name == name:
            return cfg
    raise Exception('Unknown DMLab env')


def make_dmlab_env(dmlab_cfg, mode='train'):
    """Heavily inspired by https://github.com/google-research/episodic-curiosity"""

    env_settings = dmlab_utils.create_env_settings(
        level.dmlab_level_name,
        homepath=dmlab_homepath,
        width=Const.OBSERVATION_WIDTH,
        height=Const.OBSERVATION_HEIGHT,
        seed=seed,
        main_observation=main_observation)

    env = dmlab_utils.DMLabWrapper(
        'dmlab',
        env_settings,
        action_set=get_action_set(action_set),
        main_observation=main_observation)

    # env = gym.make(atari_cfg.env_id)
    # env._max_episode_steps = atari_cfg.default_timeout
    #
    # assert 'NoFrameskip' in env.spec.id
    #
    # env = OneLifeWrapper(env)
    # env = StickyActionWrapper(env)
    # env = MaxAndSkipWrapper(env, skip=4)
    #
    # if 'Montezuma' in atari_cfg.env_id or 'Pitfall' in atari_cfg.env_id:
    #     env = AtariVisitedRoomsInfoWrapper(env)
    #
    # env = ResizeAndGrayscaleWrapper(env, ATARI_W, ATARI_H, add_channel_dim=True, area_interpolation=True)
    # env = ClipRewardWrapper(env)
    #
    # if mode == 'test':
    #     env = RenderWrapper(env)

    return env
