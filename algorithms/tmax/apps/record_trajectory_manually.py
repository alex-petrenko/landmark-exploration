import datetime
import pickle
import sys
import time
from os.path import join
from threading import Thread

from pynput.keyboard import Key, Listener

from algorithms.utils.algo_utils import main_observation
from algorithms.utils.env_wrappers import reset_with_info
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
from algorithms.topological_maps.topological_map import TopologicalMap
from utils.envs.atari import atari_utils
from utils.envs.doom import doom_utils
from utils.envs.envs import create_env
from utils.envs.generate_env_map import generate_env_map
from utils.timing import Timing
from utils.utils import log, ensure_dir_exists


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


# noinspection PyCallingNonCallable
def on_release(key):
    global current_actions
    action = key_to_action(key)
    if action is not None:
        if action in current_actions:
            current_actions.remove(action)


def record_trajectory(params, env_id):
    def make_env_func():
        e = create_env(env_id, skip_frames=False)
        e.seed(0)
        return e

    env = make_env_func()
    map_img, coord_limits = generate_env_map(make_env_func)

    env_obs, info = reset_with_info(env)
    obs = main_observation(env_obs)
    done = False

    m = TopologicalMap(obs, directed_graph=False, initial_info=info, verbose=True)

    trajectory = []
    frame = 0
    frame_repeat = 4
    action = 0

    t = Timing()

    while not done and not terminate:
        with t.timeit('one_frame'):
            env.render()

            if frame % frame_repeat == 0:
                if len(current_actions) > 0:
                    action = current_actions[-1]
                else:
                    action = 0

                trajectory.append({'obs': obs, 'action': action, 'info': info})
                m.add_landmark(obs, info, update_curr_landmark=True)

            env_obs, rew, done, info = env.step(action)
            obs = main_observation(env_obs)

        took_seconds = t.one_frame
        desired_fps = 40
        wait_seconds = (1.0 / desired_fps) - took_seconds
        wait_seconds = max(0.0, wait_seconds)
        time.sleep(wait_seconds)

        frame += 1

    env.render()
    time.sleep(0.2)

    experiment_dir = params.experiment_dir()
    trajectories_dir = ensure_dir_exists(join(experiment_dir, '.trajectories'))

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    trajectory_dir = ensure_dir_exists(join(trajectories_dir, f'traj_{timestamp}'))
    log.info('Saving to %s...', trajectory_dir)

    with open(join(trajectory_dir, 'trajectory.pickle'), 'wb') as traj_file:
        pickle.dump(trajectory, traj_file)

    m.save_checkpoint(trajectory_dir, map_img=map_img, coord_limits=coord_limits, verbose=True)

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

    status = record_trajectory(params, args.env)

    if not terminate:
        log.debug('Press ESC to exit...')
    listener_thread.join()

    return status


if __name__ == '__main__':
    sys.exit(main())
