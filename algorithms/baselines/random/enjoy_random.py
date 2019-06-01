import sys
import time

import cv2
import numpy as np

from algorithms.baselines.random.agent_random import AgentRandom
from algorithms.baselines.random.random_utils import parse_args_random
from algorithms.utils.algo_utils import main_observation, goal_observation, EPS, num_env_steps
from algorithms.utils.env_wrappers import reset_with_info
from utils.envs.envs import create_env
from utils.utils import log


def setup_histogram(agent):
    max_histogram_length = 200
    if not agent.coord_limits:
        return None

    w = (agent.coord_limits[2] - agent.coord_limits[0])
    h = (agent.coord_limits[3] - agent.coord_limits[1])
    if w > h:
        len_x = max_histogram_length
        len_y = int((h / w) * max_histogram_length)
    else:
        len_x = int((w / h) * max_histogram_length)
        len_y = max_histogram_length
    hist = np.zeros((len_x, len_y), dtype=np.int32)
    return hist


def update_coverage(agent, infos, histogram):
    for i, info in enumerate(infos):
        info = infos[i]
        if 'pos' not in info:
            continue

        agent_x, agent_y = info['pos']['agent_x'], info['pos']['agent_y']

        # Get agent coordinates normalized to [0, 1]
        dx = (agent_x - agent.coord_limits[0]) / (agent.coord_limits[2] - agent.coord_limits[0])
        dy = (agent_y - agent.coord_limits[1]) / (agent.coord_limits[3] - agent.coord_limits[1])

        # Rescale coordinates to histogram dimensions
        # Subtract eps to exclude upper bound of dx, dy
        dx = int((dx - EPS) * histogram.shape[0])
        dy = int((dy - EPS) * histogram.shape[1])

        histogram[dx, dy] += 1


last_coverage_summary = time.time()


def write_summaries(agent, histogram, env_steps, force=False):
    global last_coverage_summary

    time_since_last = time.time() - last_coverage_summary
    if time_since_last < 2 and not force:
        return

    last_coverage_summary = time.time()
    # noinspection PyProtectedMember
    agent._write_position_heatmap_summaries(
        tag='random_position_coverage', step=env_steps, histograms=[histogram],
    )


def enjoy(params, env_id, max_num_episodes=1, max_num_frames=1e10, render=False):
    def make_env_func():
        e = create_env(env_id, mode='train', skip_frames=True)
        e.seed(0)
        return e

    agent = AgentRandom(make_env_func, params.load())
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

    histogram = setup_histogram(agent)

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    for _ in range(max_num_episodes):
        env_obs, info = reset_with_info(env)
        done = False
        obs, goal_obs = main_observation(env_obs), goal_observation(env_obs)

        episode_reward = []

        while not done and not max_frames_reached(num_frames):
            start = time.time()
            if render:
                env.render()

            action = agent.best_action([obs], goals=[goal_obs], deterministic=False)
            env_obs, rew, done, info = env.step(action)
            if done:
                log.warning('Done flag is true %d, rew: %.3f, num_frames %d', done, rew, num_frames)

            update_coverage(agent, [info], histogram)

            episode_reward.append(rew)

            if num_frames % 100 == 0:
                log.info('fps: %.1f, rew: %d, done: %s, frames %d', 1.0 / (time.time() - start), rew, done, num_frames)

            write_summaries(agent, histogram, num_frames)

            num_frames += num_env_steps([info])

        if render:
            env.render()
        time.sleep(0.2)

        episode_rewards.append(sum(episode_reward))
        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        log.info(
            'Episode reward: %f, avg reward for %d episodes: %f', sum(episode_reward), len(last_episodes), avg_reward,
        )

        if max_frames_reached(num_frames):
            break

    write_summaries(agent, histogram, num_frames, force=True)

    agent.finalize()
    env.close()
    cv2.destroyAllWindows()


def main():
    args, params = parse_args_random(AgentRandom.Params)
    return enjoy(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
