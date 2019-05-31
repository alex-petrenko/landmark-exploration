import copy
import math
import time
from functools import partial

import numpy as np
import tensorflow as tf

from algorithms.agent import AgentLearner, TrainStatus
from algorithms.multi_env import MultiEnv
from algorithms.utils.algo_utils import calculate_gae, EPS, num_env_steps, main_observation, goal_observation
from algorithms.utils.encoders import make_encoder, make_encoder_with_goal, get_enc_params
from algorithms.utils.env_wrappers import main_observation_space, is_goal_based_env
from algorithms.utils.models import make_model
from algorithms.utils.tf_utils import dense, count_total_parameters, placeholder_from_space, placeholders, \
    image_summaries_rgb, summary_avg_min_max, merge_summaries
from utils.distributions import CategoricalProbabilityDistribution
from utils.envs.generate_env_map import generate_env_map
from utils.timing import Timing
from utils.utils import log, AttrDict, summaries_dir

class AgentRandom(AgentLearner):
    """Random agent (chooses actions uniformly at random)."""

    def __init__(self, make_env_func, params):
        """Initialize PPO computation graph and some auxiliary tensors."""
        super(AgentRandom, self).__init__(params)

        self.make_env_func = make_env_func
        env = make_env_func()  # we need the env to query observation shape, number of actions, etc.

        self.obs_shape = [-1] + list(main_observation_space(env).shape)
        self.ph_observations = placeholder_from_space(main_observation_space(env))
        self.ph_actions = placeholder_from_space(env.action_space)  # actions sampled from the policy
        self.ph_advantages, self.ph_returns, self.ph_old_action_probs = placeholders(None, None, None)

        env.close()

        self.objectives = None 

        summary_dir = summaries_dir(self.params.experiment_dir())
        self.summary_writer = tf.summary.FileWriter(summary_dir)

        if self.params.use_env_map:
            self.map_img, self.coord_limits = generate_env_map(make_env_func)

    def input_dict(self, buffer, start, end):  # TODO: not needed for AgentRandom?
        feed_dict = {
            self.ph_observations: buffer.obs[start:end],
            self.ph_actions: buffer.actions[start:end],
            self.ph_old_action_probs: buffer.action_probs[start:end],
            self.ph_advantages: buffer.advantages[start:end],
            self.ph_returns: buffer.returns[start:end],
        }

        return feed_dict

    def _maybe_print(self, step, env_step, avg_rewards, avg_length, fps, t):
        log.info('<====== Step %d, env step %.2fM ======>', step, env_step / 1e6)
        log.info('Avg FPS: %.1f', fps)
        log.info('Timing: %s', t)

        if math.isnan(avg_rewards) or math.isnan(avg_length):
            return

        log.info('Avg. %d episode lenght: %.3f', self.params.stats_episodes, avg_length)
        # TODO update. Just store the best of avg_reward in a class variable. Why is this a tensor?
        best_avg_reward = self.best_avg_reward.eval(session=self.session)
        log.info(
            'Avg. %d episode reward: %.3f (best: %.3f)',
            self.params.stats_episodes, avg_rewards, best_avg_reward,
        )

    def _maybe_aux_summaries(self, env_steps, avg_reward, avg_length, fps):
        self._report_basic_summaries(fps, env_steps)

        if math.isnan(avg_reward) or math.isnan(avg_length):
            # not enough data to report yet
            return

        summary = tf.Summary()
        summary.value.add(tag='0_aux/avg_reward', simple_value=float(avg_reward))
        summary.value.add(tag='0_aux/avg_length', simple_value=float(avg_length))

        # if it's not "initialized" yet, just don't report anything to tensorboard
        initial_best, best_reward = [0, 0]. # TODO: @Aleksei choose a better default?
        if best_reward != initial_best:
            summary.value.add(tag='0_aux/best_reward_ever', simple_value=float(best_reward))

        self.summary_writer.add_summary(summary, env_steps)
        self.summary_writer.flush()

    def best_action(self, observation, goals=None, deterministic=False):
        num_actions = #get num actions
        actions = np.randint((0,num_actions))
        return actions[0]

    def _train_actor(self, buffer, env_steps):
        #train actor
        return

    def _train_critic(self, buffer, env_steps):
        # train critic
        return
        
    def _train(self, buffer, env_steps):
        return

    def _learn_loop(self, multi_env):
        """Main training loop."""
        return

    def learn(self):
        status = TrainStatus.SUCCESS
        multi_env = None
        try:
            multi_env = MultiEnv(
                self.params.num_envs,
                self.params.num_workers,
                make_env_func=self.make_env_func,
                stats_episodes=self.params.stats_episodes,
            )

            self._learn_loop(multi_env)
        except (Exception, KeyboardInterrupt, SystemExit):
            log.exception('Interrupt...')
            status = TrainStatus.FAILURE
        finally:
            log.info('Closing env...')
            if multi_env is not None:
                multi_env.close()

        return status
