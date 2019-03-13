import time
import unittest
from unittest import TestCase

from algorithms.agent import AgentRandom
from algorithms.algo_utils import num_env_steps
from algorithms.multi_env import MultiEnv
from utils.envs.atari.atari_utils import make_atari_env, atari_env_by_name
from utils.envs.dmlab.dmlab_utils import make_dmlab_env, dmlab_env_by_name
from utils.envs.doom.doom_utils import make_doom_env, doom_env_by_name
from utils.timing import Timing
from utils.utils import log


def test_env_performance(test, env_type):
    t = Timing()
    with t.timeit('init'):
        env = test.make_env()
        total_num_frames, frames = 4000, 0
        agent = AgentRandom(test.make_env, {})

    with t.timeit('first_reset'):
        env.reset()

    t.reset = t.step = 1e-9
    num_resets = 0
    with t.timeit('experience'):
        while frames < total_num_frames:
            done = False

            start_reset = time.time()
            env.reset()
            t.reset += time.time() - start_reset
            num_resets += 1

            while not done and frames < total_num_frames:
                start_step = time.time()
                _, _, done, info = env.step(agent.best_action())
                t.step += time.time() - start_step
                frames += num_env_steps([info])

    fps = total_num_frames / t.experience
    log.debug('%s performance:', env_type)
    log.debug('Took %.3f sec to collect %d frames on one CPU, %.1f FPS', t.experience, total_num_frames, fps)
    log.debug('Avg. reset time %.3f s', t.reset / num_resets)
    log.debug('Timing: %s', t)

    env.close()


def test_multi_env_performance(test, env_type, num_envs, num_workers):
    t = Timing()
    with t.timeit('init'):
        multi_env = MultiEnv(num_envs, num_workers, test.make_env, stats_episodes=100)
        total_num_frames, frames = 20000, 0

    with t.timeit('first_reset'):
        multi_env.reset()

    next_print = print_step = 10000
    with t.timeit('experience'):
        while frames < total_num_frames:
            _, _, done, info = multi_env.step([0] * num_envs)
            frames += num_env_steps(info)
            if frames > next_print:
                log.info('Collected %d frames of experience...', frames)
                next_print += print_step

    fps = total_num_frames / t.experience
    log.debug('%s performance:', env_type)
    log.debug('Took %.3f sec to collect %d frames in parallel, %.1f FPS', t.experience, total_num_frames, fps)
    log.debug('Timing: %s', t)

    multi_env.close()


class TestDoom(TestCase):
    @staticmethod
    def make_env():
        return make_doom_env(doom_env_by_name('doom_maze'))

    def test_doom_env(self):
        self.assertIsNotNone(self.make_env())

    def test_doom_performance(self):
        test_env_performance(self, 'doom')

    def test_doom_performance_multi(self):
        test_multi_env_performance(self, 'doom', num_envs=128, num_workers=16)


class TestAtari(TestCase):
    @staticmethod
    def make_env():
        return make_atari_env(atari_env_by_name('atari_montezuma'))

    def test_atari_env(self):
        self.assertIsNotNone(self.make_env())

    def test_atari_performance(self):
        test_env_performance(self, 'montezuma')

    def test_atari_performance_multi(self):
        test_multi_env_performance(self, 'atari', num_envs=128, num_workers=16)


class TestDmlab(TestCase):
    @staticmethod
    def make_env():
        env = make_dmlab_env(dmlab_env_by_name('dmlab_sparse'))
        env.seed(0)
        return env

    def test_dmlab_env(self):
        self.assertIsNotNone(self.make_env())

    def skipped_test_dmlab_performance(self):
        test_env_performance(self, 'dmlab')

    def skipped_test_dmlab_performance_multi(self):
        """
        Does not work well together with other tests, because of limitations of DMLab.
        I think it does not work after VizDoom because of some conflict of graphics contexts.
        """
        test_multi_env_performance(self, 'dmlab', num_envs=128, num_workers=16)
