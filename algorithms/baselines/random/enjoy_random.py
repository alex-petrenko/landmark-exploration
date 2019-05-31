import sys
import time

import cv2

from algorithms.baselines.random.agent_random import AgentRandom
from algorithms.baselines.random.random_utils import parse_args_random  # TODO:fill
from algorithms.utils.algo_utils import main_observation, goal_observation
from algorithms.utils.env_wrappers import reset_with_info
from utils.envs.envs import create_env
from utils.utils import log


def enjoy(params, env_id, max_num_episodes=1000000, max_num_frames=1e9, fps=20):
    def make_env_func():
        e = create_env(env_id, mode='test', skip_frames=True)
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

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    for _ in range(max_num_episodes):
        env_obs, info = reset_with_info(env)
        done = False
        obs, goal_obs = main_observation(env_obs), goal_observation(env_obs)
        if goal_obs is not None:
            goal_obs_rgb = cv2.cvtColor(goal_obs, cv2.COLOR_BGR2RGB)
            cv2.imshow('goal', cv2.resize(goal_obs_rgb, (500, 500)))
            cv2.waitKey(500)

        episode_reward = []

        while not done:
            start = time.time()
            env.render()
            if fps < 1000:
                time.sleep(1.0 / fps)
            action = agent.best_action([obs], goals=[goal_obs], deterministic=False)
            env_obs, rew, done, _ = env.step(action)
            obs, goal_obs = main_observation(env_obs), goal_observation(env_obs)
            episode_reward.append(rew)
            log.info('fps: %.1f, rew: %d, done: %s', 1.0 / (time.time() - start), rew, done)

            agent._maybe_coverage_summaries(num_frames)
            agent._maybe_aux_summaries(num_frames, rew, len(episode_reward), fps)

            num_frames += 1
            if max_frames_reached(num_frames):
                break

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

    agent.finalize()
    env.close()
    cv2.destroyAllWindows()


def main():
    args, params = parse_args_random(AgentRandom.Params) # TODO: see random.params class
    return enjoy(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
