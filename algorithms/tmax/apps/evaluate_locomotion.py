import random
import sys

import numpy as np
import tensorflow as tf

from algorithms.multi_env import MultiEnv
from algorithms.tmax.agent_tmax import AgentTMAX
from algorithms.tmax.navigator import Navigator
from algorithms.utils.algo_utils import main_observation
from utils.envs.envs import create_env
from utils.timing import Timing
from utils.utils import log


def policy_step(agent, obs_prev, observations, next_target_obs, goals):
    if random.random() < 0.5:
        deterministic = False
    else:
        deterministic = True

    actions = np.zeros(len(observations), dtype=np.int32)
    obs_prev = np.array(obs_prev)
    observations = np.array(observations)
    next_target_obs = np.array(next_target_obs)

    envs_with_goal = []
    for env_i in range(len(goals)):
        if goals[env_i] is not None:
            envs_with_goal.append(env_i)

    if len(envs_with_goal) > 0:
        actions[envs_with_goal] = agent.locomotion.navigate(
            agent.session,
            obs_prev[envs_with_goal], observations[envs_with_goal], next_target_obs[envs_with_goal],
            deterministic=deterministic,
        )

    return actions


def evaluate_locomotion_agent(agent, multi_env):
    num_envs = multi_env.num_envs

    observations = main_observation(multi_env.reset())
    obs_prev = observations
    infos = multi_env.info()

    agent.tmax_mgr.initialize(observations, infos, 1)
    m = agent.tmax_mgr.dense_persistent_maps[-1]

    navigator = Navigator(agent)
    for env_i in range(num_envs):
        navigator.reset(env_i, m)

    # sample final goals
    all_targets = list(m.graph.nodes)
    if len(all_targets) > 0:
        all_targets.remove(0)

    final_goal_idx = random.sample(all_targets, num_envs)
    log.info('Goals: %r', final_goal_idx)

    # noinspection PyProtectedMember
    navigator._ensure_paths_to_goal_calculated([m] * num_envs, final_goal_idx)
    path_lengths = [0] * num_envs
    for env_i in range(num_envs):
        location, path_lenght = 0, 0
        while location != final_goal_idx[env_i]:
            location = navigator.paths[env_i][location]
            path_lenght += 1
        path_lengths[env_i] = path_lenght

    frames = 0
    next_target, next_target_d = navigator.get_next_target(
        [m] * num_envs, observations, final_goal_idx, [frames] * num_envs,
    )
    next_target_obs = [m.get_observation(t) for t in next_target]

    avg_speed = [-1] * num_envs
    success = [False] * num_envs

    t = Timing()
    while True:
        with t.timeit('frame'):
            with t.timeit('policy'):
                actions = policy_step(agent, obs_prev, observations, next_target_obs, final_goal_idx)

            with t.timeit('step'):
                env_obs, rew, done, info = multi_env.step(actions)

            obs_prev = observations
            observations = main_observation(env_obs)

            with t.timeit('navigator'):
                next_target, next_target_d = navigator.get_next_target(
                    [m] * num_envs, observations, final_goal_idx, [frames] * num_envs,
                )

            for env_i in range(num_envs):
                if final_goal_idx[env_i] is None:
                    continue

                if next_target[env_i] is None:
                    log.warning(
                        'Agent %d got lost in %d steps trying to reach %d', env_i, frames, final_goal_idx[env_i],
                    )
                    final_goal_idx[env_i] = None
                else:
                    if next_target[env_i] == final_goal_idx[env_i] and next_target_d[env_i] < 0.1:
                        success[env_i] = True
                        avg_speed[env_i] = path_lengths[env_i] / (frames + 1)
                        log.debug(
                            'Agent %d reached goal %d in %d steps, avg. speed %.3f',
                            env_i, final_goal_idx[env_i], frames, avg_speed[env_i],
                        )
                        final_goal_idx[env_i] = None

                    next_target_obs[env_i] = m.get_observation(next_target[env_i])

            frames += 1
            if frames > 5000:
                log.error('Timeout! 5000 frames was not enough to finish locomotion!')
                break

        finished = [g is None for g in final_goal_idx]
        if all(finished):
            log.info('Done!')
            break
        else:
            if frames % 10 == 0:
                frame_repeat = 4
                fps = (1.0 / t.frame) * frame_repeat * num_envs
                log.info('%d agents remaining, fps %.3f, time %s', num_envs - sum(finished), fps, t)

    return success, avg_speed


def evaluate_experiment(env_id, experiment_name, num_envs=96):
    # fixed seeds
    random.seed(0)
    np.random.seed(0)
    tf.random.set_random_seed(0)

    params = AgentTMAX.Params(experiment_name)
    params = params.load()
    params.seed = 0

    # for faster evaluation
    params.num_envs = num_envs
    params.num_workers = 32 if num_envs >= 32 else num_envs

    def make_env_func():
        e = create_env(env_id, skip_frames=True)
        e.seed(0)
        return e

    agent = AgentTMAX(make_env_func, params)
    agent.initialize()

    rate, speed = 0, -1

    multi_env = None
    try:
        multi_env = MultiEnv(
            params.num_envs,
            params.num_workers,
            make_env_func=make_env_func,
            stats_episodes=params.stats_episodes,
        )

        success, avg_speed = evaluate_locomotion_agent(agent, multi_env)

        log.info('Finished evaluating experiment %s', experiment_name)
        rate = np.mean(success)
        speed = -1
        avg_speed = [s for s in avg_speed if s > 0]
        if len(avg_speed) > 0:
            speed = np.mean(avg_speed)

        log.info('Success rate %.1f%%, avg. speed %.2f edges/frame', rate * 100, speed)

    except (Exception, KeyboardInterrupt, SystemExit):
        log.exception('Interrupt...')
    finally:
        log.info('Closing env...')
        if multi_env is not None:
            multi_env.close()

    agent.finalize()
    return rate, speed


def evaluate_locomotion():
    experiments = (
        # ('doom_textured_super_sparse', 'doom_textured_super_sparse-tmax_v035-64filt'),
        # ('doom_textured_super_sparse', 'doom_textured_super_sparse-tmax_v035-gamma-0998'),
        # ('doom_maze_no_goal', 'doom_maze_no_goal-tmax_v035_dist_expl'),
        # ('doom_maze_no_goal', 'doom_maze_no_goal-tmax_v035_no_spars'),

        ('doom_textured_super_sparse_v2', 'doom_textured_super_sparse_v2_trajectory'),
    )

    t = Timing()

    with t.timeit('evaluation'):
        results = {}
        for experiment in experiments:
            env_id, exp_name = experiment
            rate, speed = evaluate_experiment(env_id, exp_name)
            results[exp_name] = (rate, speed)

    log.info('Evaluation completed, took %s', t)
    rates, speeds = [], []
    for exp_name, r in results.items():
        rate, speed = r
        log.info('%s: success_rate: %.1f%%, avg_speed %.3f', exp_name, rate * 100, speed)
        rates.append(rate)
        speeds.append(speed)

    log.info('Average across experiments: success %.1f%%, speed: %.3f', np.mean(rates) * 100, np.mean(speeds))

    return 0


def main():
    status = evaluate_locomotion()
    return status


if __name__ == '__main__':
    sys.exit(main())
