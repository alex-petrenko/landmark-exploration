import multiprocessing

from utils.img_utils import crop_map_image
from utils.utils import log, AttrDict


def _generate_env_map_worker(make_env_func, return_dict):
    map_img = coord_limits = None
    env = make_env_func()

    try:
        if env.unwrapped.coord_limits and hasattr(env.unwrapped, 'get_automap_buffer'):
            from vizdoom import ScreenResolution
            env.unwrapped.show_automap = True
            env.unwrapped.screen_w = 800
            env.unwrapped.screen_h = 600
            env.unwrapped.screen_resolution = ScreenResolution.RES_800X600
            env.reset()
            env.unwrapped.game.advance_action()
            map_img = env.unwrapped.get_automap_buffer()
            map_img = crop_map_image(map_img)
            coord_limits = env.unwrapped.coord_limits
    except AttributeError as exc:
        log.warning(f'Could not get map image from env, exception: {exc}')
    finally:
        env.close()

    return_dict['map_img'] = map_img
    return_dict['coord_limits'] = coord_limits


def generate_env_map(make_env_func):
    """
    Currently only Doom environments support this.
    We have to initialize the env instance in a separate process because otherwise Doom overrides the signal handler
    and we cannot do things like KeyboardInterrupt anympore.
    """
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    p = multiprocessing.Process(target=_generate_env_map_worker, args=(make_env_func, return_dict))
    p.start()
    p.join()

    return_dict = AttrDict(return_dict)
    return return_dict.map_img, return_dict.coord_limits
