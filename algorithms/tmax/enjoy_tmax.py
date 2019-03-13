import sys
import time
from threading import Thread

from pynput.keyboard import Key, Listener

from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
from utils.envs.envs import create_env
from utils.utils import log

pause = False
terminate = False


def on_press(key):
    global pause
    if key == Key.space:
        pause = not pause


def on_release(key):
    if key == Key.esc:
        global terminate
        terminate = True
        return False


def enjoy(params, env_id, max_num_episodes=1000, max_num_frames=None, fps=1000):
    def make_env_func():
        e = create_env(env_id, mode='test')
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
        obs, done = env.reset(), False
        episode_reward, episode_frames = 0, 0

        if agent.tmax_mgr.initialized:
            agent.tmax_mgr.update([obs], [True], verbose=True)
        else:
            agent.tmax_mgr.initialize([obs])

        start_episode = time.time()
        while not done and not terminate and not max_frames_reached(num_frames):
            env.render()
            if fps < 1000:
                time.sleep(1.0 / fps)

            if pause:
                continue

            action = agent.best_action([obs], deterministic=False)
            obs, rew, done, _ = env.step(action)

            if not done:
                bonus = agent.tmax_mgr.update([obs], [done], verbose=True)
                bonus = bonus[0]
                if bonus > 0:
                    log.info('Bonus %.3f received', bonus)
                if abs(rew) >= 0.01:
                    log.info('Reward %.3f received', rew)

            episode_reward += rew

            num_frames += 1
            episode_frames += 1

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
    return 0


def main():
    # start keypress listener (to pause/resume execution or exit)
    def start_listener():
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()

    listener_thread = Thread(target=start_listener)
    listener_thread.start()

    args, params = parse_args_tmax(AgentTMAX.Params)
    status = enjoy(params, args.env)

    log.debug('Press ESC to exit...')
    listener_thread.join()

    return status


if __name__ == '__main__':
    sys.exit(main())
