import sys
import time
from threading import Thread

from pynput.keyboard import Key, Listener

from algorithms.algo_utils import main_observation
from algorithms.env_wrappers import reset_with_info
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
from algorithms.topological_maps.topological_map import TopologicalMap, get_position, get_angle
from utils.envs.atari import atari_utils
from utils.envs.doom import doom_utils
from utils.envs.envs import create_env
from utils.envs.generate_env_map import generate_env_map
from utils.utils import log, model_dir

add_landmark = False
terminate = False
current_actions = []
key_to_action = None


# noinspection PyCallingNonCallable
def on_press(key):
    if key == Key.esc:
        global terminate
        terminate = True
        return False

    global current_actions
    action = key_to_action(key)
    if action is not None:
        if action not in current_actions:
            current_actions.append(action)

    global add_landmark
    if key == Key.enter:
        add_landmark = True


# noinspection PyCallingNonCallable
def on_release(key):
    global current_actions
    action = key_to_action(key)
    if action is not None:
        if action in current_actions:
            current_actions.remove(action)


def build_graph(params, env_id, max_num_episodes=1000):
    def make_env_func():
        e = create_env(env_id, mode='test', skip_frames=False)
        e.seed(0)
        return e

    map_img, coord_limits = generate_env_map(make_env_func)
    env = make_env_func()

    m = None

    for _ in range(max_num_episodes):
        env_obs, info = reset_with_info(env)
        obs = main_observation(env_obs)
        done = False

        if m is None:
            m = TopologicalMap(obs, directed_graph=False, initial_info=info, verbose=True)

        while not done and not terminate:
            env.render()

            if len(current_actions) > 0:
                action = current_actions[-1]
            else:
                action = 0

            env_obs, rew, done, info = env.step(action)
            obs = main_observation(env_obs)

            global add_landmark
            if add_landmark:
                # noinspection PyProtectedMember
                new_idx = m._add_new_node(obs=obs, pos=get_position(info), angle=get_angle(info))
                log.info('Added landmark idx %d', new_idx)
                add_landmark = False

        if terminate:
            break
        else:
            env.render()
            time.sleep(0.2)

    checkpoint_dir = model_dir(params.experiment_dir())
    m.save_checkpoint(checkpoint_dir, map_img=map_img, coord_limits=coord_limits, verbose=True)
    log.debug('Set breakpoint here to edit graph edges before saving...')

    log.info('Saving to %s...', checkpoint_dir)
    m.save_checkpoint(checkpoint_dir, map_img=map_img, coord_limits=coord_limits, verbose=True)

    env.close()
    return 0


def main():
    args, params = parse_args_tmax(AgentTMAX.Params)
    env_id = args.env

    global key_to_action
    if 'dmlab' in env_id:
        from utils.envs.dmlab import play_dmlab
        key_to_action = play_dmlab.key_to_action
    elif 'atari' in env_id:
        key_to_action = atari_utils.key_to_action
    elif 'doom' in env_id:
        key_to_action = doom_utils.key_to_action
    else:
        raise Exception('Unknown env')

    # start keypress listener (to pause/resume execution or exit)
    def start_listener():
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    listener_thread = Thread(target=start_listener)
    listener_thread.start()

    status = build_graph(params, args.env)

    if not terminate:
        log.debug('Press ESC to exit...')
    listener_thread.join()

    return status


if __name__ == '__main__':
    sys.exit(main())
