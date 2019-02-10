import time
from unittest import TestCase

from algorithms.agent import AgentRandom
from utils.envs.atari.atari_utils import make_atari_env, atari_env_by_name
from utils.envs.doom.doom_utils import make_doom_env, doom_env_by_name
from utils.utils import log


def test_env_performance(test, env_type):
    env = test.make_env()
    total_num_frames, frames = 1000, 0
    agent = AgentRandom(test.make_env, {})

    start = time.time()
    while frames < total_num_frames:
        done = False
        env.reset()
        while not done and frames < total_num_frames:
            _, _, done, _ = env.step(agent.best_action())
            frames += 1
    total_time = time.time() - start

    fps = total_num_frames / total_time
    log.debug('%s performance:', env_type)
    log.debug('Took %.3f sec to collect %d frames on one CPU, %.1f FPS', total_time, total_num_frames, fps * 4)

    env.close()


class TestDoom(TestCase):
    @staticmethod
    def make_env():
        return make_doom_env(doom_env_by_name('doom_maze'))

    def test_doom_env(self):
        self.assertIsNotNone(self.make_env())

    def test_doom_performance(self):
        test_env_performance(self, 'doom')


class TestAtari(TestCase):
    @staticmethod
    def make_env():
        return make_atari_env(atari_env_by_name('atari_montezuma'))

    def test_atari_env(self):
        self.assertIsNotNone(self.make_env())

    def test_atari_performance(self):
        test_env_performance(self, 'montezuma')

