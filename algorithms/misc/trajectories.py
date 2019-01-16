import sys
import time
from os.path import join

import cv2
import numpy as np

from algorithms.arguments import parse_model, parse_args
from utils.envs.envs import create_env
from utils.utils import log, data_dir, ensure_dir_exists, remove_if_exists


def get_trajectories(params, env_id, agent_cls, max_num_episodes=1000000, fps=1500):
    def make_env_func():
        e = create_env(env_id, mode='traj')
        e.seed(0)
        return e

    agent = agent_cls(make_env_func, params.load())
    env = make_env_func()

    # this helps with screen recording
    pause_at_the_beginning = False
    if pause_at_the_beginning:
        env.render()
        log.info('Press any key to start...')
        cv2.waitKey()

    agent.initialize()

    traj_dir = join(data_dir(params.experiment_dir()), 'trajectories')
    ensure_dir_exists(traj_dir)

    max_episodes_strlen = len(str(max_num_episodes))

    episode_rewards = []
    for i in range(max_num_episodes):
        obs, done = env.reset(), False
        episode_reward = 0

        ep_obs, ep_actions, ep_rewards, ep_dones = [], [], [], []

        while not done:
            ep_obs.append(obs)

            start = time.time()
            # env.render()
            if fps < 1000:
                time.sleep(1.0 / fps)

            action = agent.best_action(obs, deterministic=False)
            ep_actions.append(action)

            obs, rew, done, _ = env.step(action)
            ep_rewards.append(rew)
            ep_dones.append(done)
            episode_reward += rew

            log.info('Actual fps: %.1f', 1.0 / (time.time() - start))

        # env.render()

        episode_rewards.append(episode_reward)
        last_episodes = episode_rewards[-100:]
        avg_reward = sum(last_episodes) / len(last_episodes)
        log.info(
            'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
        )

        traj_filename = 'ep_' + str(i).zfill(max_episodes_strlen) + '_' + params.filename_prefix() + 'traj.npz'
        traj_filename = join(traj_dir, traj_filename)
        remove_if_exists(traj_filename)

        np.savez(traj_filename, obs=ep_obs, action=ep_actions, reward=ep_rewards, done=ep_dones)

    agent.finalize()
    env.close()
    return 0


def main():
    agent_cls = parse_model()
    args, params = parse_args('doom_basic', '', agent_cls.Params)
    return get_trajectories(params, args.env, agent_cls)


if __name__ == '__main__':
    sys.exit(main())
