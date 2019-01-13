import threading
import time
from multiprocessing import Process, JoinableQueue
from enum import Enum
import copy
from queue import Queue

import numpy as np

from algorithms.algo_utils import list_to_string
from utils.utils import log, AttrDict


class StepType(Enum):
    REAL = 1
    IMAGINED = 2


class _MultiEnvWorker:
    """Helper class for the MultiEnv."""

    def __init__(self, env_indices, make_env_func, use_multiprocessing):
        self._verbose = False

        self.env_indices = env_indices

        self.envs = []
        self.initial_obs = []

        log.info('Initializing envs %s...', list_to_string(env_indices))
        for i in env_indices:
            env = make_env_func()
            env.seed(i)
            self.initial_obs.append(env.reset())
            self.envs.append(env)

        self.imagined_envs = None

        if use_multiprocessing:
            self.step_queue, self.result_queue = JoinableQueue(), JoinableQueue()
            self.process = Process(target=self.start, daemon=True)
        else:
            log.debug('Not using multiprocessing!')
            self.step_queue, self.result_queue = Queue(), Queue()
            # shouldn't be named "process" here, but who cares
            self.process = threading.Thread(target=self.start)

        self.process.start()

    def start(self):
        timing = AttrDict({'copying': 0, 'prediction': 0})

        while True:
            actions, step_type = self.step_queue.get()
            if actions is None:  # stop signal
                for e in self.envs:
                    e.close()
                if self.imagined_envs is not None:
                    for imagined_env in self.imagined_envs:
                        imagined_env.close()
                self.step_queue.task_done()
                log.info('Stop worker %s...', list_to_string(self.env_indices))
                break

            if step_type == StepType.REAL:
                envs = self.envs
                if self.imagined_envs is not None:
                    for imagined_env in self.imagined_envs:
                        imagined_env.close()
                self.imagined_envs = None
            else:  # step_type == StepType.IMAGINED:

                if self.imagined_envs is None:
                    # initializing new prediction, let's report timing for the previous one
                    if timing.prediction > 0 and self._verbose:
                        log.debug(
                            'Multi-env copy took %.6f s, prediction took %.6f s',
                            timing.copying, timing.prediction,
                        )

                    timing.prediction = 0
                    timing.copying = time.time()

                    self.imagined_envs = []
                    # we expect a list of actions for every environment in this worker (list of lists)
                    assert len(actions) == len(self.envs)
                    for env_idx in range(len(actions)):
                        for _ in actions[env_idx]:
                            imagined_env = copy.deepcopy(self.envs[env_idx])
                            self.imagined_envs.append(imagined_env)
                    timing.copying = time.time() - timing.copying

                envs = self.imagined_envs
                actions = np.asarray(actions).flatten()

            assert len(envs) == len(actions)

            # Collect obs, reward, and 'done' for each env (discard info)
            prediction_start = time.time()
            results = [env.step(action) for env, action in zip(envs, actions)]

            # pack results per-env
            results = np.split(np.array(results), len(self.envs))

            if step_type == StepType.IMAGINED:
                timing.prediction += time.time() - prediction_start

            # If this is a real step and the env is done, reset
            if step_type == StepType.REAL:
                for i, result in enumerate(results):
                    obs, reward, done, info = result[0]
                    if done:
                        obs = self.envs[i].reset()
                    results[i] = (obs, reward, done, info)  # collapse dimension of size 1

            self.result_queue.put(results)
            self.step_queue.task_done()


class MultiEnv:
    """Run multiple gym-compatible environments in parallel, keeping more or less the same interface."""

    def __init__(self, num_envs, num_workers, make_env_func, stats_episodes, use_multiprocessing=True):
        self._verbose = False

        if num_workers > num_envs or num_envs % num_workers != 0:
            raise Exception('num_envs should be a multiple of num_workers')

        self.num_envs = num_envs
        self.num_workers = num_workers
        self.workers = []

        envs = np.split(np.arange(num_envs), num_workers)
        self.workers = [
            _MultiEnvWorker(envs[i].tolist(), make_env_func, use_multiprocessing)
            for i in range(num_workers)
        ]

        self.action_space = self.workers[0].envs[0].action_space
        self.observation_space = self.workers[0].envs[0].observation_space

        self.curr_episode_reward = [0] * num_envs
        self.episode_rewards = [[] for _ in range(num_envs)]

        self.curr_episode_duration = [0] * num_envs
        self.episode_lengths = [[] for _ in range(num_envs)]

        self.stats_episodes = stats_episodes

    def initial_obs(self):
        obs = []
        for w in self.workers:
            for o in w.initial_obs:
                obs.append(o)
        return obs

    def step(self, actions):
        """Obviously, returns vectors of obs, rewards, dones instead of usual single values."""
        assert len(actions) == self.num_envs
        actions = np.split(np.array(actions), self.num_workers)
        for worker, action_tuple in zip(self.workers, actions):
            worker.step_queue.put((action_tuple, StepType.REAL))

        results = []
        for worker in self.workers:
            worker.step_queue.join()
            results_per_worker = worker.result_queue.get()
            assert len(results_per_worker) == self.num_envs // self.num_workers
            for result in results_per_worker:
                results.append(result)
            worker.result_queue.task_done()

        observations, rewards, dones, infos = zip(*results)

        for i in range(self.num_envs):
            self.curr_episode_reward[i] += rewards[i]

            step_len = 1
            if infos[i] is not None and 'num_frames' in infos[i]:
                step_len = infos[i]['num_frames']

            self.curr_episode_duration[i] += step_len

            if dones[i]:
                self._update_episode_stats(self.episode_rewards[i], self.curr_episode_reward[i])
                self.curr_episode_reward[i] = 0
                self._update_episode_stats(self.episode_lengths[i], self.curr_episode_duration[i])
                self.curr_episode_duration[i] = 0

        return observations, rewards, dones, infos

    def predict(self, imagined_action_lists):
        start = time.time()
        assert len(imagined_action_lists) == self.num_envs
        imagined_action_lists = np.split(np.array(imagined_action_lists), self.num_workers)
        for worker, imagined_action_list in zip(self.workers, imagined_action_lists):
            worker.step_queue.put((imagined_action_list, StepType.IMAGINED))

        observations = []
        rewards = []
        dones = []
        for worker in self.workers:
            worker.step_queue.join()
            results_per_worker = worker.result_queue.get()
            assert len(results_per_worker) == len(imagined_action_list)
            for result in results_per_worker:
                o, r, d, _ = zip(*result)
                observations.append(o)
                rewards.append(r)
                dones.append(d)
            worker.result_queue.task_done()

        if self._verbose:
            log.debug('Prediction step took %.4f s', time.time() - start)
        return observations, rewards, dones

    def close(self):
        log.info('Stopping multi env...')
        for worker in self.workers:
            worker.step_queue.put((None, StepType.REAL))  # terminate
            worker.step_queue.join()
            worker.process.join()

    def _update_episode_stats(self, episode_stats, curr_episode_data):
        episode_stats_target_size = 2 * (1 + self.stats_episodes // self.num_envs)
        episode_stats.append(curr_episode_data)
        if len(episode_stats) > episode_stats_target_size * 2:
            del episode_stats[:episode_stats_target_size]

    def _calc_episode_stats(self, episode_data, n):
        n_per_env = 1 + n // self.num_envs  # number of last episodes to use from every environment

        avg_value = 0
        mean_of_n_episodes = 0
        for i in range(self.num_envs):
            last_values = episode_data[i][-n_per_env:]
            if len(last_values) <= 0:
                continue

            avg_value += np.sum(last_values)
            mean_of_n_episodes += len(last_values)

        # to prevent reporting statistics too early
        if mean_of_n_episodes < max(n, self.num_envs):
            return np.nan

        return avg_value / mean_of_n_episodes

    def calc_avg_rewards(self, n):
        return self._calc_episode_stats(self.episode_rewards, n)

    def calc_avg_episode_lengths(self, n):
        return self._calc_episode_stats(self.episode_lengths, n)

    def stats_num_episodes(self):
        worker_lenghts = [len(r) for r in self.episode_rewards]
        return sum(worker_lenghts)
