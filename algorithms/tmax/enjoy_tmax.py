import random
import sys
import time
from threading import Thread

import cv2
from pynput.keyboard import Key, Listener, KeyCode

from algorithms.utils.algo_utils import main_observation, goal_observation, EPS
from algorithms.utils.env_wrappers import reset_with_info
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax, TmaxMode
from algorithms.topological_maps.topological_map import TopologicalMap
from utils.envs.atari import atari_utils
from utils.envs.doom import doom_utils
from utils.envs.envs import create_env
from utils.timing import Timing
from utils.utils import log


class PolicyType:
    RANDOM, AGENT, LOCOMOTION, PLAYER = range(4)
    KEY_CHARS = {RANDOM: 'r', AGENT: 'a', LOCOMOTION: 'l', PLAYER: 'p'}
    KEYS = {t: KeyCode.from_char(c) for t, c in KEY_CHARS.items()}


persistent_map = None
current_landmark = None

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
    if key == Key.pause:
        pause = not pause

    global current_actions
    action = key_to_action(key)
    if action is not None:
        if action not in current_actions:
            current_actions.append(action)

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

    global persistent_map
    if agent.params.persistent_map_checkpoint is not None:
        persistent_map = TopologicalMap.create_empty()
        persistent_map.maybe_load_checkpoint(agent.params.persistent_map_checkpoint)

    global current_landmark

    episode_rewards = []
    num_frames = 0

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    for _ in range(max_num_episodes):
        env_obs, info = reset_with_info(env)
        done = False

        obs, goal_obs = main_observation(env_obs), goal_observation(env_obs)
        prev_obs = obs
        if current_landmark is None:
            current_landmark = obs

        if goal_obs is not None:
            goal_obs_rgb = cv2.cvtColor(goal_obs, cv2.COLOR_BGR2RGB)
            cv2.imshow('goal', cv2.resize(goal_obs_rgb, (500, 500)))
            cv2.waitKey(500)

        episode_reward, episode_frames = 0, 0

        if agent.tmax_mgr.initialized:
            _, _ = agent.tmax_mgr.update([obs], [obs], [0], [True], [info], num_frames, verbose=True)
        else:
            agent.tmax_mgr.initialize([obs], [info], env_steps=0)
            persistent_map = agent.tmax_mgr.dense_persistent_maps[-1]
            sparse_persistent_map = agent.tmax_mgr.sparse_persistent_maps[-1]
            log.debug('Num landmarks in sparse map: %d', sparse_persistent_map.num_landmarks())

        # TODO
        agent.curiosity.initialized = True
        agent.tmax_mgr.mode[0] = TmaxMode.EXPLORATION
        agent.tmax_mgr.locomotion_final_targets[0] = None
        agent.tmax_mgr.locomotion_targets[0] = None

        start_episode = time.time()
        t = Timing()
        while not done and not terminate and not max_frames_reached(num_frames):
            with t.timeit('one_frame'):
                env.render()
                cv2.waitKey(1)  # to prevent window from fading

                if pause:
                    time.sleep(0.01)
                    continue

                if len(current_actions) > 0:
                    # key combinations are not handled, but this is purely for testing
                    action = current_actions[-1]
                else:
                    action = 0

                if policy_type == PolicyType.PLAYER:
                    pass
                elif policy_type == PolicyType.RANDOM:
                    action = env.action_space.sample()
                elif policy_type == PolicyType.AGENT:
                    agent.tmax_mgr.mode[0] = TmaxMode.EXPLORATION
                    action, *_ = agent.policy_step([prev_obs], [obs], [goal_obs], None, None)
                    action = action[0]
                elif policy_type == PolicyType.LOCOMOTION:
                    agent.tmax_mgr.mode[0] = TmaxMode.LOCOMOTION
                    action, _, _ = agent.loco_actor_critic.invoke(
                        agent.session, [obs], [current_landmark], None, None, [1.0],
                    )
                    action = action[0]

                env_obs, rew, done, info = env.step(action)
                next_obs, goal_obs = main_observation(env_obs), goal_observation(env_obs)

                _, _ = agent.tmax_mgr.update(
                    [obs], [next_obs], [rew], [done], [info], num_frames, t, verbose=True,
                )

                prev_obs = obs
                obs = next_obs

                episode_reward += rew

                num_frames += 1
                episode_frames += 1

            took_seconds = t.one_frame
            desired_fps = 15
            wait_seconds = (1.0 / desired_fps) - took_seconds
            wait_seconds = max(0.0, wait_seconds)
            if wait_seconds > EPS:
                time.sleep(wait_seconds)

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
