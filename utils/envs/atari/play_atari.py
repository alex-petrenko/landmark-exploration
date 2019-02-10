import sys
import time
from threading import Thread

import cv2
from gym.envs.atari.atari_env import ACTION_MEANING
from pynput.keyboard import Key, Listener

from utils.envs.envs import create_env
from utils.utils import log


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


terminate = False
current_actions = []


def on_press(key):
    if key == Key.esc:
        global terminate
        terminate = True
        return False

    global current_actions
    if key in action_table:
        if action_table[key] not in current_actions:
            current_actions.append(action_table[key])


def on_release(key):
    global current_actions
    if key in action_table:
        if action_table[key] in current_actions:
            current_actions.remove(action_table[key])


def main():
    # start keypress listener
    def start_listener():
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    listener_thread = Thread(target=start_listener)
    listener_thread.start()

    env = create_env('atari_montezuma')

    env.reset()
    done = False
    episode_reward = 0
    fps = 30
    while not done and not terminate:
        atari_img = env.render(mode='rgb_array')
        atari_img = cv2.cvtColor(atari_img, cv2.COLOR_BGR2RGB)
        atari_img_big = cv2.resize(atari_img, (420, 420), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('atari', atari_img_big)
        cv2.waitKey(1000 // fps)

        if len(current_actions) > 0:
            # key combinations are not handled, but this is purely for testing
            action_name = current_actions[-1]
        else:
            action_name = 'NOOP'

        action = action_name_to_action(action_name)
        obs, reward, done, info = env.step(action)
        episode_reward += reward

        if reward != 0:
            log.debug('Reward received: %.3f', reward)

    log.info('Episode reward %.3f', episode_reward)
    if not terminate:
        log.info('Press ESC to exit...')
    listener_thread.join()
    log.info('Done')

    env.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
