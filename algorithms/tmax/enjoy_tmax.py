import sys
import time

import cv2

from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.tmax_utils import parse_args_tmax
from algorithms.tmax.topological_map import TopologicalMap
from utils.envs.envs import create_env
from utils.utils import log


def enjoy(params, env_id, max_num_episodes=1000000, max_num_frames=None, fps=10):
    def make_env_func():
        e = create_env(env_id, mode='test')
        e.seed(0)
        return e

    agent = AgentTMAX(make_env_func, params.load())
    env = make_env_func()

    # this helps with screen recording
    pause_at_the_beginning = False
    if pause_at_the_beginning:
        env.render()
        log.info('Press any key to start...')
        cv2.waitKey()

    agent.initialize()

    episode_rewards = []
    num_frames = 0

    graph = None

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    for _ in range(max_num_episodes):
        obs, done = env.reset(), False
        episode_reward, episode_frames = 0, 0
        if graph is None:
            graph = TopologicalMap(obs, verbose=True)

        start_episode = time.time()
        while not done:
            env.render()
            if fps < 1000:
                time.sleep(1.0 / fps)
            action = agent.best_action(obs, deterministic=False)
            obs, rew, done, _ = env.step(action)

            bonus = agent.update_maps([graph], [obs], [done], verbose=True)
            bonus = bonus[0]
            if bonus > 0:
                log.info('Bonus %.3f received', bonus)

            episode_reward += rew

            num_frames += 1
            episode_frames += 1
            if max_frames_reached(num_frames):
                break

        env.render()
        log.info('Actual fps: %.1f', 1.0 / (time.time() - start_episode))
        time.sleep(0.2)

        episode_rewards.append(episode_reward)
        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        log.info(
            'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
        )

        if max_frames_reached(num_frames):
            break

    agent.finalize()
    env.close()
    return 0


def main():
    args, params = parse_args_tmax(AgentTMAX.Params)
    return enjoy(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
