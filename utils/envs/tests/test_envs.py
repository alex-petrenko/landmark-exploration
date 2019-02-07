import time
from unittest import TestCase

from algorithms.agent import AgentRandom
from utils.envs.doom.doom_utils import make_doom_env, env_by_name
from utils.utils import log


class TestDoom(TestCase):
    @staticmethod
    def make_env():
        return make_doom_env(env_by_name('maze'))

    def test_doom_env(self):
        self.assertIsNotNone(self.make_env())

    def test_doom_performance(self):
        env = self.make_env()
        total_num_frames, frames = 1000, 0
        agent = AgentRandom(self.make_env, {})

        start = time.time()
        while frames < total_num_frames:
            done = False
            env.reset()
            while not done and frames < total_num_frames:
                _, _, done, _ = env.step(agent.best_action())
                frames += 1
        total_time = time.time() - start

        fps = total_num_frames / total_time
        log.debug('Took %.3f sec to collect %d frames on one CPU, %.1f FPS', total_time, total_num_frames, fps * 4)

        env.close()
