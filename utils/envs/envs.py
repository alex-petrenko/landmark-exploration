from utils.envs.atari.atari_utils import make_atari_env, atari_env_by_name

from utils.envs.doom.doom_utils import make_doom_env, doom_env_by_name


def create_env(env, **kwargs):
    """Expected names are: doom_maze, atari_montezuma, etc."""

    if env.startswith('doom_'):
        return make_doom_env(doom_env_by_name(env), **kwargs)
    elif env.startswith('atari_'):
        return make_atari_env(atari_env_by_name(env), **kwargs)
    elif env.startswith('dmlab_'):
        from utils.envs.dmlab.dmlab_utils import make_dmlab_env, dmlab_env_by_name
        return make_dmlab_env(dmlab_env_by_name(env), **kwargs)
    else:
        raise Exception('Unsupported env {0}'.format(env))
