import sys
import time
from threading import Thread

import cv2
from pynput.keyboard import Key, Listener, KeyCode

from algorithms.algo_utils import main_observation, goal_observation
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
from utils.envs.atari import atari_utils
from utils.envs.doom import doom_utils
from utils.envs.envs import create_env
from utils.utils import log


class PolicyType:
    RANDOM, AGENT, LOCOMOTION, PLAYER = range(4)
    KEY_CHARS = {RANDOM: 'r', AGENT: 'a', LOCOMOTION: 'l', PLAYER: 'p'}
    KEYS = {t: KeyCode.from_char(c) for t, c in KEY_CHARS.items()}


store_landmark = True
pause = False
terminate = False
policy_type = PolicyType.AGENT
current_actions = []
key_to_action = None


# noinspection PyCallingNonCallable
def on_press(key):
    if key == Key.esc:
        global terminate
        terminate = True
        return False

    global pause
    if key == Key.space:
        pause = not pause

    global current_actions
    action = key_to_action(key)
    if action is not None:
        if action not in current_actions:
            current_actions.append(action)

    global store_landmark
    if key == Key.enter:
        store_landmark = True

    global policy_type
    for t, k in PolicyType.KEYS.items():
        if key == k:
            policy_type = t
            log.info('Switch to policy %d (%r)', t, k)


# noinspection PyCallingNonCallable
def on_release(key):
    global current_actions
    action = key_to_action(key)
    if action is not None:
        if action in current_actions:
            current_actions.remove(action)


def enjoy(params, env_id, max_num_episodes=1000, max_num_frames=None, show_automap=False):
    def make_env_func():
        e = create_env(env_id, mode='test', show_automap=show_automap)
        e.seed(0)
        return e

    params = params.load()
    params.num_envs = 1  # during execution we're only using one env
    agent = AgentTMAX(make_env_func, params)
    env = make_env_func()

    agent.initialize()

    episode_rewards = []
    num_frames = 0

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    for _ in range(max_num_episodes):
        env_obs, done = env.reset(), False
        info = env.unwrapped.get_info()

        current_landmark = env_obs
        obs, goal_obs = main_observation(env_obs), goal_observation(env_obs)
        if goal_obs is not None:
            goal_obs_rgb = cv2.cvtColor(goal_obs, cv2.COLOR_BGR2RGB)
            cv2.imshow('goal', cv2.resize(goal_obs_rgb, (500, 500)))
            cv2.waitKey(500)

        episode_reward, episode_frames = 0, 0

        if agent.tmax_mgr.initialized:
            agent.tmax_mgr.update([obs], [goal_obs], [True], [info], verbose=True)
        else:
            agent.tmax_mgr.initialize([obs], [info])

        start_episode = time.time()
        while not done and not terminate and not max_frames_reached(num_frames):
            env.render()
            cv2.waitKey(1)  # to prevent window from fading
            if show_automap:
                automap = env.unwrapped.get_automap_buffer() # (600, 800, 3)
                if automap is not None:
                    cv2.namedWindow('Landmark Map')
                    for landmark_pos in agent.tmax_mgr.episodic_maps[0].positions:
                        if 'agent_x' in landmark_pos:
                            x = int(landmark_pos['agent_x'])
                            y = -int(landmark_pos['agent_y'])
                            a = int(landmark_pos['agent_a'])
                            automap = cv2.circle(automap, (y, x), 1, (0,0,0), thickness=-1)
                    for landmark_pos in agent.tmax_mgr.maps[0].positions:
                        if 'agent_x' in landmark_pos:
                            x = int(landmark_pos['agent_x'])
                            y = -int(landmark_pos['agent_y'])
                            a = int(landmark_pos['agent_a'])
                            automap = cv2.circle(automap, (y, x), 1, (0,0,0), thickness=-1)
                    cv2.imshow('Landmark Map', automap)
                    cv2.waitKey(1)

            if pause:
                time.sleep(0.01)
                continue

            if len(current_actions) > 0:
                # key combinations are not handled, but this is purely for testing
                action = current_actions[-1]
            else:
                action = 0

            if policy_type == PolicyType.RANDOM:
                action = env.action_space.sample()
            elif policy_type == PolicyType.AGENT:
                action = agent.policy_step([obs], [goal_obs], None, None, is_bootstrap=False)[0]
            elif policy_type == PolicyType.LOCOMOTION:
                action = agent.locomotion.navigate(agent.session, [obs], [current_landmark], deterministic=False)[0]
                log.info('Locomotion action %d', action)

            env_obs, rew, done, info = env.step(action)
            obs, goal_obs = main_observation(env_obs), goal_observation(env_obs)

            if not done:
<<<<<<< HEAD
                bonus = agent.tmax_mgr.update([obs], [goal_obs], [done], [info], verbose=True)
=======
                bonus, _, _ = agent.tmax_mgr.update([obs], [goal_obs], [done], verbose=True)
>>>>>>> master
                bonus = bonus[0]
                if bonus > 0:
                    log.info('Bonus %.3f received', bonus)
                if abs(rew) >= 0.01:
                    log.info('Reward %.3f received', rew)

            episode_reward += rew

            num_frames += 1
            episode_frames += 1

            global store_landmark
            if store_landmark:
                log.warning('Store new landmark!')
                current_landmark = obs
                store_landmark = False

        env.render()
        log.info('Actual fps: %.1f', episode_frames / (time.time() - start_episode))
        time.sleep(0.2)

        episode_rewards.append(episode_reward)
        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        log.info(
            'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
        )

        if max_frames_reached(num_frames) or terminate:
            break

    agent.finalize()
    env.close()
    cv2.destroyAllWindows()

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

    try:
        show_map = args.show_automap
    except AttributeError:
        show_map = False

    # start keypress listener (to pause/resume execution or exit)
    def start_listener():
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    listener_thread = Thread(target=start_listener)
    listener_thread.start()

    status = enjoy(params, args.env, show_automap=show_map)

    log.debug('Press ESC to exit...')
    listener_thread.join()

    return status


if __name__ == '__main__':
    sys.exit(main())
