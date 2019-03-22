import copy
import time

from algorithms.algo_utils import num_env_steps, main_observation, goal_observation
from algorithms.baselines.ppo.agent_ppo import AgentPPO, PPOBuffer
from algorithms.curiosity.icm.icm import IntrinsicCuriosityModule
from algorithms.curiosity.reachability_curiosity.reachability_curiosity import ReachabilityCuriosityModule
from algorithms.env_wrappers import main_observation_space
from algorithms.tf_utils import placeholder_from_space
from algorithms.trajectory import TrajectoryBuffer
from utils.timing import Timing


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
    class Params(AgentPPO.Params):
        """Hyperparams for curious PPO"""
        def __init__(self, experiment_name):
            super(AgentCuriousPPO.Params, self).__init__(experiment_name)

            self.curiosity_type = 'icm'  # icm or reachability

            # icm parameters
            self.cm_beta = 0.5
            self.cm_lr_scale = 10.0
            self.clip_bonus = 0.1
            self.prediction_bonus_coeff = 0.05  # scaling factor for prediction bonus vs env rewards
            self.forward_fc = 256

            # episodic curiosity parameters
            self.reachable_threshold = 8  # num. of frames between obs, such that one is reachable from the other
            self.unreachable_threshold = 24  # num. of frames between obs, such that one is unreachable from the other
            self.reachability_target_buffer_size = 100000  # target number of training examples to store
            self.reachability_train_epochs = 10
            self.reachability_batch_size = 128
            self.reachability_bootstrap = 1000000
            self.reachability_train_interval = 500000

            self.new_landmark_threshold = 0.9  # condition for considering current observation a "new landmark"
            self.loop_closure_threshold = 0.7  # condition for graph loop closure (finding new edge)
            self.map_expansion_reward = 0.2  # reward for finding new vertex

        @staticmethod
        def filename_prefix():
            return 'curious_ppo_'

    def __init__(self, make_env_func, params):
        super(AgentCuriousPPO, self).__init__(make_env_func, params)

        env = self.make_env_func()  # we need it to query observation shape, number of actions, etc.
        self.ph_next_observations = placeholder_from_space(main_observation_space(env))

        if self.params.curiosity_type == 'icm':
            # create graph for curiosity module (ICM)
            self.curiosity = IntrinsicCuriosityModule(
                env, self.ph_observations, self.ph_next_observations, self.ph_actions, params.forward_fc, params,
            )
        elif self.params.curiosity_type == 'reachability':
            self.curiosity = ReachabilityCuriosityModule(env, params)
        else:
            raise Exception(f'Curiosity type {self.params.curiosity_type} not supported')

    def _train_with_curiosity(self, step, buffer, env_steps, timing):
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
                    actions, action_probs, values = self.actor_critic.invoke(self.session, obs, goals)

                    # wait for all the workers to complete an environment step
                    env_obs, rewards, dones, infos = multi_env.step(actions)
                    next_obs, new_goals = main_observation(env_obs), goal_observation(env_obs)

                    trajectory_buffer.add(obs, actions, dones)

                    # calculate curiosity bonus
                    bonuses = self.curiosity.generate_bonus_rewards(
                        self.session, obs, next_obs, actions, dones, infos,
                    )
                    rewards += bonuses

                    # add experience from environment to the current buffer
                    buffer.add(obs, next_obs, actions, action_probs, rewards, dones, values, goals)
                    goals = new_goals
                    obs = next_obs

                    num_steps += num_env_steps(infos)

                # last step values are required for TD-return calculation
                _, _, values = self.actor_critic.invoke(self.session, obs, goals)
                buffer.values.append(values)

            env_steps += num_steps

            # calculate discounted returns and GAE
            buffer.finalize_batch(self.params.gamma, self.params.gae_lambda)

            # update actor and critic and CM
            with timing.timeit('train'):
                step = self._train_with_curiosity(step, buffer, env_steps, timing)

            avg_reward = multi_env.calc_avg_rewards(n=self.params.stats_episodes)
            avg_length = multi_env.calc_avg_episode_lengths(n=self.params.stats_episodes)
            fps = num_steps / (time.time() - batch_start)

            self._maybe_print(step, env_steps, avg_reward, avg_length, fps, timing)
            self._maybe_aux_summaries(env_steps, avg_reward, avg_length)
            self._maybe_update_avg_reward(avg_reward, multi_env.stats_num_episodes())

            trajectory_buffer.reset_trajectories()
