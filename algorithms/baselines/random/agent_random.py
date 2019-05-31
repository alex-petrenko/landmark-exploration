import time

import math
import numpy as np
import tensorflow as tf

from algorithms.agent import AgentLearner
from utils.envs.generate_env_map import generate_env_map
from utils.utils import summaries_dir


class AgentRandom(AgentLearner):
    """Random agent (chooses actions uniformly at random)."""
    class Params(AgentLearner.AgentParams):
        """Hyperparams for curious PPO"""

        def __init__(self, experiment_name):
            # calling all parent constructors
            super(AgentRandom.Params, self).__init__(experiment_name)

            self.action_repeat_chance = 0.95

        @staticmethod
        def filename_prefix():
            return 'random_'

        def load(self):
            return self

    def __init__(self, make_env_func, params):
        """Initialize PPO computation graph and some auxiliary tensors."""
        super(AgentRandom, self).__init__(params)

        self.make_env_func = make_env_func
        env = make_env_func()
        self.action_space = env.action_space
        env.close()

        self.objectives = None

        self.last_action = None
        self.best_reward = None

        summary_dir = summaries_dir(self.params.experiment_dir())
        self.summary_writer = tf.summary.FileWriter(summary_dir)

        if self.params.use_env_map:
            self.map_img, self.coord_limits = generate_env_map(make_env_func)

    def _maybe_aux_summaries(self, env_steps, reward, length, fps):
        self._report_basic_summaries(fps, env_steps)

        if math.isnan(reward) or math.isnan(length):
            # not enough data to report yet
            return

        summary = tf.Summary()
        summary.value.add(tag='0_aux/avg_reward', simple_value=float(reward))
        summary.value.add(tag='0_aux/avg_length', simple_value=float(length))

        if not self.best_reward or self.best_reward < reward:
            self.best_reward = reward
        if self.best_reward:
            summary.value.add(tag='0_aux/best_reward_ever', simple_value=float(self.best_reward))

        self.summary_writer.add_summary(summary, env_steps)
        self.summary_writer.flush()

    def best_action(self, observation, goals=None, deterministic=False):
        # Repeat last action with probability action_repeat_chance
        if self.last_action and np.random.randint(0, 100) > 100 * self.params.action_repeat_chance:
            return self.last_action
        action = self.action_space.sample()
        self.last_action = action
        return action

    def _train_actor(self, buffer, env_steps):
        pass

    def _train_critic(self, buffer, env_steps):
        pass
        
    def _train(self, buffer, env_steps):
        pass

    def _learn_loop(self, multi_env):
        """Main training loop."""
        pass

    def learn(self):
        pass
