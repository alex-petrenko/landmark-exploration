import numpy as np
import random
import sys
from threading import Thread

from gym.envs.atari.atari_env import ACTION_MEANING
from pynput.keyboard import Key, Listener, KeyCode

from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
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


class PolicyType:
    RANDOM, IDLE_RANDOM, LOCOMOTION, PLAYER = range(4)
    KEY_CHARS = {RANDOM: 'r', IDLE_RANDOM: 'i', LOCOMOTION: 'l', PLAYER: 'p'}
    KEYS = {t: KeyCode.from_char(c) for t, c in KEY_CHARS.items()}


store_landmark = True
terminate = False
policy_type = PolicyType.PLAYER
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

    global store_landmark
    if key == Key.enter:
        store_landmark = True

    global policy_type
    for t, k in PolicyType.KEYS.items():
        if key == k:
            policy_type = t
            log.info('Switch to policy %d (%r)', t, k)


def on_release(key):
    global current_actions
    if key in action_table:
        if action_table[key] in current_actions:
            current_actions.remove(action_table[key])


def play_and_visualize(params, env_id):
    def make_env_func():
        e = create_env(env_id, mode='test')
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

    obs = current_landmark = env.reset()
    done = False
    episode_reward = 0

    frame = 0
    current_landmark_frame = frame
    idle_frames = 0
    deliberate_actions = 0

    while not done and not terminate:
        env.render()

        if len(current_actions) > 0:
            # key combinations are not handled, but this is purely for testing
            action_name = current_actions[-1]
        else:
            action_name = 'NOOP'

        if policy_type == PolicyType.RANDOM:
            action = env.action_space.sample()
            idle_frames = 0
        elif policy_type == PolicyType.IDLE_RANDOM:
            if idle_frames > 0 and random.random() < 0.97:
                action = 0  # NOOP
                idle_frames -= 1
                if idle_frames % 10 == 0:
                    log.info('Idle frames %d, deliberate actions %d', idle_frames, deliberate_actions)
            else:
                action = env.action_space.sample()
                deliberate_actions += 1
                if random.random() < 0.015:
                    idle_frames = np.random.randint(1, 400)
        elif policy_type == PolicyType.LOCOMOTION:
            action = agent.locomotion.navigate(agent.session, [obs], [current_landmark], deterministic=True)
        else:
            action = action_name_to_action(action_name)
            idle_frames = 0

        obs, reward, done, info = env.step(action)
        episode_reward += reward
        frame += 1

        global store_landmark
        if store_landmark:
            log.warning('Store new landmark!')
            current_landmark = obs
            current_landmark_frame = frame
            store_landmark = False

        if frame % 1 == 0:
            distances = agent.reachability.distances(
                agent.session, [current_landmark, obs], [obs, obs],
            )
            log.info(
                'Distance: to %.3f self %.3f, frames %d',
                distances[0], distances[1], frame - current_landmark_frame,
            )

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
    args, params = parse_args_tmax(AgentTMAX.Params)
    return play_and_visualize(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
