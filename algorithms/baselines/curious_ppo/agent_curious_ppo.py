import copy
import random
import time

import numpy as np

from algorithms.curiosity.compression.etc import ExplorationThroughCompression
from algorithms.curiosity.rnd.rnd import RandomNetworkDistillation
from algorithms.utils.algo_utils import num_env_steps, main_observation, goal_observation
from algorithms.baselines.ppo.agent_ppo import AgentPPO, PPOBuffer
from algorithms.curiosity.ecr.ecr import ECRModule
from algorithms.curiosity.ecr_map.ecr_map import ECRMapModule
from algorithms.curiosity.icm.icm import IntrinsicCuriosityModule
from algorithms.utils.env_wrappers import main_observation_space
from algorithms.utils.tf_utils import placeholder_from_space
from algorithms.utils.trajectory import TrajectoryBuffer
from utils.timing import Timing
from utils.utils import log


class CuriousPPOBuffer(PPOBuffer):
    def __init__(self):
        super(CuriousPPOBuffer, self).__init__()
        self.next_obs = None

    def reset(self):
        super(CuriousPPOBuffer, self).reset()
        self.next_obs = []

    # noinspection PyMethodOverriding
    def add(self, obs, next_obs, actions, action_probs, rewards, dones, values, goals=None):
        """Append one-step data to the current batch of observations."""
        args = copy.copy(locals())
        super(CuriousPPOBuffer, self)._add_args(args)


class AgentCuriousPPO(AgentPPO):
    """PPO with a curiosity module (ICM or RND)"""
    class Params(
        AgentPPO.Params,
        ECRModule.Params,  # find "episodic curiosity" params here
        ECRMapModule.Params,  # find "episodic curiosity" params here
        IntrinsicCuriosityModule.Params,  # find "ICM" params here
        RandomNetworkDistillation.Params,
        ExplorationThroughCompression.Params,
    ):
        """Hyperparams for curious PPO"""

        def __init__(self, experiment_name):
            # calling all parent constructors
            AgentPPO.Params.__init__(self, experiment_name)
            ECRModule.Params.__init__(self)
            ECRMapModule.Params.__init__(self)
            IntrinsicCuriosityModule.Params.__init__(self)
            RandomNetworkDistillation.Params.__init__(self)
            ExplorationThroughCompression.Params.__init__(self)

            self.curiosity_type = 'icm'  # icm or ecr or ecr_map
            self.random_exploration = False
            self.action_repeat_chance = 0.95

            self.extrinsic_reward_coeff = 1.0

            self.graceful_episode_termination = False

        @staticmethod
        def filename_prefix():
            return 'curious_ppo_'

    def __init__(self, make_env_func, params):
        super(AgentCuriousPPO, self).__init__(make_env_func, params)

        env = self.make_env_func()  # we need it to query observation shape, number of actions, etc.
        self.ph_next_observations = placeholder_from_space(main_observation_space(env))
        self.num_actions = env.action_space.n
        env.close()

        if self.params.curiosity_type == 'icm':
            # create graph for curiosity module (ICM)
            self.curiosity = IntrinsicCuriosityModule(
                env, self.ph_observations, self.ph_next_observations, self.ph_actions, params.forward_fc, params,
            )
        elif self.params.curiosity_type == 'ecr':
            self.curiosity = ECRModule(env, params)
        elif self.params.curiosity_type == 'ecr_map':
            self.curiosity = ECRMapModule(env, params)
        elif self.params.curiosity_type == 'rnd':
            self.curiosity = RandomNetworkDistillation(env, self.ph_observations, self)
        elif self.params.curiosity_type == 'etc':
            self.curiosity = ExplorationThroughCompression(env, self.ph_observations, self)
        else:
            raise Exception(f'Curiosity type {self.params.curiosity_type} not supported')

        self.previous_actions = np.random.randint(0, self.num_actions, self.params.num_envs)

    def initialize(self):
        super().initialize()
        self.curiosity.initialize(self.session)

    def _policy_step(self, obs, goals):
        if self.params.random_exploration:
            actions = self.previous_actions

            for i in range(len(obs)):
                if random.random() > self.params.action_repeat_chance:
                    action = np.random.randint(0, self.num_actions)
                    actions[i] = action

            action_probs = np.ones(self.params.num_envs)
            values = np.zeros(self.params.num_envs)

            self.previous_actions = actions
        else:
            actions, action_probs, values = self.actor_critic.invoke(self.session, obs, goals)

        return actions, action_probs, values

    def _train_with_curiosity(self, step, buffer, env_steps, timing):
        if not self.params.random_exploration:
            if self.curiosity.is_initialized():
                with timing.timeit('train_actor'):
                    step = self._train_actor(buffer, env_steps)
                with timing.timeit('train_critic'):
                    self._train_critic(buffer, env_steps)

            with timing.timeit('train_curiosity'):
                self.curiosity.train(buffer, env_steps, agent=self)

        return step

    def _learn_loop(self, multi_env):
        """Main training loop."""
        # env_steps used in tensorboard (and thus, our results)
        # actor_step used as global step for training
        step, env_steps = self.session.run([self.actor_step, self.total_env_steps])

        env_obs = multi_env.reset()
        obs, goals = main_observation(env_obs), goal_observation(env_obs)

        buffer = CuriousPPOBuffer()
        trajectory_buffer = TrajectoryBuffer(self.params.num_envs)
        self.curiosity.set_trajectory_buffer(trajectory_buffer)

        def end_of_training(s, es):
            return s >= self.params.train_for_steps or es > self.params.train_for_env_steps

        while not end_of_training(step, env_steps):
            timing = Timing()
            num_steps = 0
            batch_start = time.time()

            buffer.reset()

            with timing.timeit('experience'):
                # collecting experience
                for rollout_step in range(self.params.rollout):
                    actions, action_probs, values = self._policy_step(obs, goals)

                    # wait for all the workers to complete an environment step
                    env_obs, rewards, dones, infos = multi_env.step(actions)

                    if self.params.graceful_episode_termination:
                        rewards = list(rewards)
                        for i in range(self.params.num_envs):
                            if dones[i] and infos[i].get('prev') is not None:
                                if infos[i]['prev'].get('terminated_by_timer', False):
                                    log.info('Env %d terminated by timer', i)
                                    rewards[i] += values[i]

                    if not self.params.random_exploration:
                        trajectory_buffer.add(obs, actions, infos, dones)

                    next_obs, new_goals = main_observation(env_obs), goal_observation(env_obs)

                    # calculate curiosity bonus
                    with timing.add_time('curiosity'):
                        if not self.params.random_exploration:
                            bonuses = self.curiosity.generate_bonus_rewards(
                                self.session, obs, next_obs, actions, dones, infos,
                            )
                            rewards = self.params.extrinsic_reward_coeff * np.array(rewards) + bonuses

                    # add experience from environment to the current buffer
                    buffer.add(obs, next_obs, actions, action_probs, rewards, dones, values, goals)

                    obs, goals = next_obs, new_goals
                    self.process_infos(infos)
                    num_steps += num_env_steps(infos)

                # last step values are required for TD-return calculation
                _, _, values = self._policy_step(obs, goals)
                buffer.values.append(values)

            env_steps += num_steps

            # calculate discounted returns and GAE
            buffer.finalize_batch(self.params.gamma, self.params.gae_lambda)

            # update actor and critic and CM
            with timing.timeit('train'):
                step = self._train_with_curiosity(step, buffer, env_steps, timing)

            avg_reward = multi_env.calc_avg_rewards(n=self.params.stats_episodes)
            avg_length = multi_env.calc_avg_episode_lengths(n=self.params.stats_episodes)

            self._maybe_update_avg_reward(avg_reward, multi_env.stats_num_episodes())
            self._maybe_trajectory_summaries(trajectory_buffer, env_steps)
            self._maybe_coverage_summaries(env_steps)
            self.curiosity.additional_summaries(
                env_steps, self.summary_writer, self.params.stats_episodes,
                map_img=self.map_img, coord_limits=self.coord_limits,
            )

            trajectory_buffer.reset_trajectories()

            fps = num_steps / (time.time() - batch_start)
            self._maybe_print(step, env_steps, avg_reward, avg_length, fps, timing)
            self._maybe_aux_summaries(env_steps, avg_reward, avg_length, fps)
