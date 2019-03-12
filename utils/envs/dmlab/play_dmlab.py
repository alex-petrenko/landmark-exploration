import sys
from threading import Thread

from pynput.keyboard import Key, Listener

from utils.envs.dmlab.dmlab_utils import make_dmlab_env, dmlab_env_by_name
from utils.utils import log

action_table = {
    Key.up: 1,
    Key.down: 2,
    Key.left: 3,
    Key.right: 4,
}

current_actions = []
terminate = False


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


def play():
    env = make_dmlab_env(dmlab_env_by_name('dmlab_sparse'), mode='test')
    env.seed(0)
    env.reset()

    # start keypress listener
    def start_listener():
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    listener_thread = Thread(target=start_listener)
    listener_thread.start()

    done = False
    episode_reward = 0

    frame = 0

    while not terminate:
        if done:
            env.reset()

        env.render()

        if len(current_actions) > 0:
            # key combinations are not handled, but this is purely for testing
            action = current_actions[-1]
        else:
            action = 0

        obs, reward, done, info = env.step(action)
        episode_reward += reward
        frame += 1

        if reward != 0:
            log.debug('Reward received: %.3f', reward)

    env.render()

    log.info('Episode reward %.3f', episode_reward)
    if not terminate:
        log.info('Press ESC to exit...')
    listener_thread.join()
    log.info('Done')

    env.close()
    return 0


def main():
    return play()


if __name__ == '__main__':
    sys.exit(main())
