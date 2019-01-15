import sys
import time
from os.path import join
import zipfile

import cv2

from algorithms.baselines.ppo.agent_ppo import AgentPPO
from algorithms.baselines.ppo.ppo_utils import parse_args_ppo
from utils.envs.envs import create_env
from utils.utils import log, data_dir, experiment_dir


def enjoy(params, env_id, max_num_episodes=1000000, fps=1500):
    def make_env_func():
        e = create_env(env_id, mode='test')
        e.seed(0)
        return e

    agent = AgentPPO(make_env_func, params.load())
    env = make_env_func()

    # this helps with screen recording
    pause_at_the_beginning = False
    if pause_at_the_beginning:
        env.render()
        log.info('Press any key to start...')
        cv2.waitKey()

    agent.initialize()

    traj_filename = params.filename_prefix() + 'trajectories.npz'
    traj_dir = join(data_dir(experiment_dir()), traj_filename)

    with open(traj_dir, "wb") as trajectories:
        episode_rewards = []
        for _ in range(max_num_episodes):
            obs, done = env.reset(), False
            episode_reward = 0

            while not done:
                start = time.time()
                env.render()
                if fps < 1000:
                    time.sleep(1.0 / fps)
                action = agent.best_action(obs, deterministic=False)
                obs, rew, done, _ = env.step(action)
                episode_reward += rew

                log.info('Actual fps: %.1f', 1.0 / (time.time() - start))

            env.render()
            time.sleep(0.2)

            episode_rewards.append(episode_reward)
            last_episodes = episode_rewards[-100:]
            avg_reward = sum(last_episodes) / len(last_episodes)
            log.info(
                'Episode reward: %f, avg reward for %d episodes: %f', episode_reward, len(last_episodes), avg_reward,
            )

    agent.finalize()
    env.close()
    return 0


def main():
    args, params = parse_args_ppo(AgentPPO.Params)
    return enjoy(params, args.env)


if __name__ == '__main__':
    sys.exit(main())
