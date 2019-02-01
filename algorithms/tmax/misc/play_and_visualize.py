import sys
import time
from threading import Thread

import cv2
from pynput.keyboard import Listener, Key

from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
from utils.envs.envs import create_env
from utils.utils import log


store_landmark = True
terminate = False


def on_press(key):
    global store_landmark
    if key == Key.space:
        store_landmark = True


def on_release(key):
    if key == Key.esc:
        global terminate
        terminate = True
        return False


def play_and_visualize(params, env_id, verbose=False):
    def make_env_func():
        e = create_env(env_id, mode='test', human_input=True)
        e.seed(0)
        return e

    # start keypress listener
    def start_listener():
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    listener_thread = Thread(target=start_listener)
    listener_thread.start()

    agent = AgentTMAX(make_env_func, params.load())
    agent.initialize()

    env = make_env_func()
    current_landmark = env.reset()
    frame = 0
    current_landmark_frame = frame

    env_doom = env.unwrapped

    done = False
    while not done and not terminate:
        obs, reward, done, _ = env.step(0)
        frame += 1

        global store_landmark
        if store_landmark:
            log.warning('Store new landmark!')
            current_landmark = obs
            current_landmark_frame = frame
            store_landmark = False

        reachability_probs = agent.reachability.get_reachability(agent.session, [current_landmark], [obs])
        log.info('Reachability: %.3f frames %d', reachability_probs[0], frame - current_landmark_frame)

        if verbose:
            log.info('Action: %r', env_doom.game.get_last_action())
            log.info('Reward: %.4f', reward)

        obs_big = cv2.resize(obs, (420, 420), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('agent_observations', obs_big)
        cv2.waitKey(1)

    time.sleep(0.3)

    agent.finalize()
    env.close()

    log.debug('Press ESC to exit...')
    listener_thread.join()

    log.info('Done')
    return


def main():
    args, params = parse_args_tmax(AgentTMAX.Params)
    return play_and_visualize(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
