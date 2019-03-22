import copy
import math
import time
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from algorithms.agent import AgentLearner, TrainStatus
from algorithms.algo_utils import calculate_gae, EPS, num_env_steps, main_observation, goal_observation
from algorithms.encoders import make_encoder, make_encoder_with_goal
from algorithms.env_wrappers import main_observation_space, is_goal_based_env
from algorithms.models import make_model
from algorithms.multi_env import MultiEnv
from algorithms.tf_utils import dense, count_total_parameters, placeholder_from_space, placeholders, \
    image_summaries_rgb, summary_avg_min_max, merge_summaries
from utils.distributions import CategoricalProbabilityDistribution
from utils.timing import Timing
from utils.utils import log, AttrDict, summaries_dir


class ActorCritic:
    def __init__(self, env, ph_observations, params):
        self.ph_observations = ph_observations

        num_actions = env.action_space.n
        obs_space = env.observation_space

        # Goal observation
        self.ph_goal_obs = None
        self.is_goal_env = is_goal_based_env(env)
        if self.is_goal_env:
            # goal obs has the same shape as main obs
            self.ph_goal_obs = placeholder_from_space(main_observation_space(env))

        make_encoder_func = make_encoder_with_goal if self.is_goal_env else make_encoder

        regularizer = None  # don't use L2 regularization

        # actor computation graph
        # use actor encoder as main observation encoder (including landmarks, etc.)
        if self.is_goal_env:
            actor_encoder = make_encoder_func(self.ph_observations, self.ph_goal_obs, obs_space, regularizer,
                                              params, name='act_enc')
        else:
            actor_encoder = make_encoder_func(self.ph_observations, obs_space, regularizer, params, name='act_enc')

        actor_model = make_model(actor_encoder.encoded_input, regularizer, params, 'act_mdl')

        actions_fc = dense(actor_model.latent, params.model_fc_size // 2, regularizer)
        action_logits = tf.contrib.layers.fully_connected(actions_fc, num_actions, activation_fn=None)
        self.best_action_deterministic = tf.argmax(action_logits, axis=1)
        self.actions_distribution = CategoricalProbabilityDistribution(action_logits)
        self.act = self.actions_distribution.sample()
        self.action_prob = self.actions_distribution.probability(self.act)

        # critic computation graph
        if self.is_goal_env:
            value_encoder = make_encoder_func(self.ph_observations, self.ph_goal_obs, obs_space, regularizer,
                                              params, 'val_enc')
        else:
            value_encoder = make_encoder_func(self.ph_observations, obs_space, regularizer, params, 'val_enc')
        value_model = make_model(value_encoder.encoded_input, regularizer, params, 'val_mdl')

        value_fc = dense(value_model.latent, params.model_fc_size // 2, regularizer)
        self.value = tf.squeeze(tf.contrib.layers.fully_connected(value_fc, 1, activation_fn=None), axis=[1])

        log.info('Total parameters in the model: %d', count_total_parameters())

    def invoke(self, session, observations, goals=None, deterministic=False):
        # TODO RECURRENT
        ops = [
            self.best_action_deterministic if deterministic else self.act,
            self.action_prob,
            self.value,
        ]
        feed_dict = {self.ph_observations: observations}
        if self.is_goal_env:  # add goal input if it is given
            feed_dict[self.ph_goal_obs] = goals
        actions, action_prob, values = session.run(ops, feed_dict=feed_dict)
        return actions, action_prob, values

    def best_action(self, session, observations, goals=None, deterministic=False):
        feed_dict = {self.ph_observations: observations}
        if self.is_goal_env:  # add goal input if it is given
            feed_dict[self.ph_goal_obs] = goals
        actions = session.run(
            self.best_action_deterministic if deterministic else self.act,
            feed_dict=feed_dict,
        )
        return actions


class PPOBuffer:
    def __init__(self):
        self.obs = self.actions = self.action_probs = self.rewards = self.dones = self.values = None
        self.advantages = self.returns = self.goals = None

    def reset(self):
        self.obs, self.actions, self.action_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []
        self.goals = []
        self.advantages = self.returns = None

    def _add_args(self, args):
        for arg_name, arg_value in args.items():
            if arg_name in self.__dict__ and arg_value is not None:
                self.__dict__[arg_name].append(arg_value)

    def add(self, obs, actions, action_probs, rewards, dones, values, goals=None):
        args = copy.copy(locals())
        self._add_args(args)

    def finalize_batch(self, gamma, gae_lambda):
        # convert everything in the buffer into numpy arrays
        for item, x in self.__dict__.items():
            if x is None:
                continue
            self.__dict__[item] = np.asarray(x)  # preserve existing array type (e.g. uint8 for images)

        # calculate discounted returns and GAE
        self.advantages, self.returns = calculate_gae(self.rewards, self.dones, self.values, gamma, gae_lambda)

        # values vector has one extra last value that we don't need
        self.values = self.values[:-1]
        assert self.values.shape == self.advantages.shape

        num_transitions = self.obs.shape[0] * self.obs.shape[1]
        for item, x in self.__dict__.items():
            if x is None or x.size == 0:
                continue

            data_shape = x.shape[2:]
            # collapse [num_batches, batch_size] into one dimension
            self.__dict__[item] = x.reshape((num_transitions,) + data_shape)

    def shuffle(self):
        """Shuffle all buffers in-place with the same random seed."""
        rng_state = np.random.get_state()

        for x in self.__dict__.values():
            if x is None or x.size == 0:
                continue

            np.random.set_state(rng_state)
            np.random.shuffle(x)

    def __len__(self):
        return len(self.obs)


class AgentPPO(AgentLearner):
    """Agent based on PPO algorithm."""

    class Params(AgentLearner.AgentParams):
        """Hyperparams for the algorithm and the training process."""

        def __init__(self, experiment_name):
            """Default parameter values set in ctor."""
            super(AgentPPO.Params, self).__init__(experiment_name)

            self.gamma = 0.99  # future reward discount
            self.gae_lambda = 0.8
            self.rollout = 64
            self.num_envs = 192  # number of environments to collect the experience from
            self.num_workers = 16  # number of workers used to run the environments

            # actor-critic (encoders and models)
            self.image_enc_name = 'convnet_84px'
            self.model_fc_layers = 1
            self.model_fc_size = 256
            self.model_recurrent = False
            self.rnn_rollout = 16

            # ppo-specific
            self.ppo_clip_ratio = 1.1  # we use clip(x, e, 1/e) instead of clip(x, 1+e, 1-e) in the paper
            self.target_kl = 0.03
            self.batch_size = 512
            self.ppo_epochs = 10

            # components of the loss function
            self.initial_entropy_loss_coeff = 0.1
            self.min_entropy_loss_coeff = 0.002

            # training process
            self.learning_rate = 1e-4
            self.train_for_steps = self.train_for_env_steps = 10 * 1000 * 1000 * 1000
            self.use_gpu = True
            self.initial_save_rate = 1000

        @staticmethod
        def filename_prefix():
            return 'ppo_'

    def __init__(self, make_env_func, params):
        """Initialize PPO computation graph and some auxiliary tensors."""
        super(AgentPPO, self).__init__(params)

        self.actor_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='actor_step')
        self.critic_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='critic_step')

        self.make_env_func = make_env_func
        env = make_env_func()  # we need the env to query observation shape, number of actions, etc.

        self.obs_shape = [-1] + list(main_observation_space(env).shape)
        self.ph_observations = placeholder_from_space(main_observation_space(env))
        self.ph_actions = placeholder_from_space(env.action_space)  # actions sampled from the policy
        self.ph_advantages, self.ph_returns, self.ph_old_action_probs = placeholders(None, None, None)

        self.actor_critic = ActorCritic(env, self.ph_observations, self.params)

        env.close()

        self.objectives = self.add_ppo_objectives(
            self.actor_critic,
            self.ph_actions, self.ph_old_action_probs, self.ph_advantages, self.ph_returns,
            self.params,
            self.actor_step,
        )

        # optimizers
        actor_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='actor_opt')
        self.train_actor = actor_opt.minimize(self.objectives.actor_loss, global_step=self.actor_step)

        critic_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='critic_opt')
        self.train_critic = critic_opt.minimize(self.objectives.critic_loss, global_step=self.critic_step)

        self.add_ppo_summaries()

        summary_dir = summaries_dir(self.params.experiment_dir())
        self.summary_writer = tf.summary.FileWriter(summary_dir)
        self.actor_summaries = merge_summaries(collections=['actor'])
        self.critic_summaries = merge_summaries(collections=['critic'])

    def input_dict(self, buffer, start, end):
        # Most placeholders are in AgentPPO, so input dict is here
        feed_dict = {
            self.ph_observations: buffer.obs[start:end],
            self.ph_actions: buffer.actions[start:end],
            self.ph_old_action_probs: buffer.action_probs[start:end],
            self.ph_advantages: buffer.advantages[start:end],
            self.ph_returns: buffer.returns[start:end],
        }
        if self.actor_critic.is_goal_env:
            feed_dict[self.actor_critic.ph_goal_obs] = buffer.goals[start:end]  # TODO: move ph_goals_obs to PPO?

        return feed_dict

    @staticmethod
    def add_ppo_objectives(actor_critic, actions, old_action_probs, advantages, returns, params, step):
        action_probs = actor_critic.actions_distribution.probability(actions)
        prob_ratio = action_probs / old_action_probs  # pi / pi_old

        clip_ratio = params.ppo_clip_ratio
        clipped_advantages = tf.where(advantages > 0, advantages * clip_ratio, advantages / clip_ratio)

        clipped = tf.logical_or(prob_ratio > clip_ratio, prob_ratio < 1.0 / clip_ratio)
        clipped = tf.cast(clipped, tf.float32)

        # PPO policy gradient loss
        ppo_loss = tf.reduce_mean(-tf.minimum(prob_ratio * advantages, clipped_advantages))

        # penalize for inaccurate value estimation
        value_loss = tf.reduce_mean(tf.square(returns - actor_critic.value))

        # penalize the agent for being "too sure" about it's actions (to prevent converging to the suboptimal local
        # minimum too soon)
        entropy_losses = actor_critic.actions_distribution.entropy()

        # make sure entropy is maximized only for state-action pairs with non-clipped advantage
        entropy_losses = (1.0 - clipped) * entropy_losses
        entropy_loss = -tf.reduce_mean(entropy_losses)
        entropy_loss_coeff = tf.train.exponential_decay(
            params.initial_entropy_loss_coeff, tf.cast(step, tf.float32), 10.0, 0.95, staircase=True,
        )
        entropy_loss_coeff = tf.maximum(entropy_loss_coeff, params.min_entropy_loss_coeff)
        entropy_loss = entropy_loss_coeff * entropy_loss

        # auxiliary quantities (for tensorboard, logging, early stopping)
        log_p_old = tf.log(old_action_probs + EPS)
        log_p = tf.log(action_probs + EPS)
        sample_kl = tf.reduce_mean(log_p_old - log_p)
        sample_entropy = tf.reduce_mean(-log_p)
        clipped_fraction = tf.reduce_mean(clipped)

        # only use entropy bonus if the policy is not close to max entropy
        max_entropy = actor_critic.actions_distribution.max_entropy()
        entropy_loss = tf.cond(sample_entropy > 0.8 * max_entropy, lambda: 0.0, lambda: entropy_loss)

        # final losses to optimize
        actor_loss = ppo_loss + entropy_loss
        critic_loss = value_loss

        return AttrDict(locals())

    def add_ppo_summaries(self):
        obj = self.objectives

        # summaries for the agent and the training process
        with tf.name_scope('obs_summaries'):
            image_summaries_rgb(self.ph_observations, collections=['actor'])

        with tf.name_scope('actor'):
            summary_avg_min_max('returns', self.ph_returns, collections=['actor'])
            summary_avg_min_max('adv', self.ph_advantages, collections=['actor'])

            actor_scalar = partial(tf.summary.scalar, collections=['actor'])
            actor_scalar('action_avg', tf.reduce_mean(tf.to_float(self.actor_critic.act)))
            actor_scalar('selected_action_avg', tf.reduce_mean(tf.to_float(self.ph_actions)))

            actor_scalar('entropy', tf.reduce_mean(self.actor_critic.actions_distribution.entropy()))
            actor_scalar('entropy_coeff', obj.entropy_loss_coeff)

            actor_scalar('actor_training_steps', self.actor_step)

            with tf.name_scope('ppo'):
                actor_scalar('sample_kl', obj.sample_kl)
                actor_scalar('sample_entropy', obj.sample_entropy)
                actor_scalar('clipped_fraction', obj.clipped_fraction)

            with tf.name_scope('losses'):
                actor_scalar('action_loss', obj.ppo_loss)
                actor_scalar('entropy_loss', obj.entropy_loss)
                actor_scalar('actor_loss', obj.actor_loss)

        with tf.name_scope('critic'):
            critic_scalar = partial(tf.summary.scalar, collections=['critic'])
            critic_scalar('value', tf.reduce_mean(self.actor_critic.value))
            critic_scalar('value_loss', obj.value_loss)
            critic_scalar('critic_training_steps', self.critic_step)

    def _maybe_print(self, step, env_step, avg_rewards, avg_length, fps, t):
        log.info('<====== Step %d, env step %.2fM ======>', step, env_step / 1e6)
        log.info('Avg FPS: %.1f', fps)
        log.info('Timing: %s', t)

        if math.isnan(avg_rewards) or math.isnan(avg_length):
            return

        log.info('Avg. %d episode lenght: %.3f', self.params.stats_episodes, avg_length)
        best_avg_reward = self.best_avg_reward.eval(session=self.session)
        log.info(
            'Avg. %d episode reward: %.3f (best: %.3f)',
            self.params.stats_episodes, avg_rewards, best_avg_reward,
        )

    def _maybe_aux_summaries(self, env_steps, avg_reward, avg_length):
        if math.isnan(avg_reward) or math.isnan(avg_length):
            # not enough data to report yet
            return

        summary = tf.Summary()
        summary.value.add(tag='0_aux/avg_reward', simple_value=float(avg_reward))
        summary.value.add(tag='0_aux/avg_length', simple_value=float(avg_length))

        # if it's not "initialized" yet, just don't report anything to tensorboard
        initial_best, best_reward = self.session.run([self.initial_best_avg_reward, self.best_avg_reward])
        if best_reward != initial_best:
            summary.value.add(tag='0_aux/best_reward_ever', simple_value=float(best_reward))

        self.summary_writer.add_summary(summary, env_steps)
        self.summary_writer.flush()

    def best_action(self, observation, goals=None, deterministic=False):
        actions = self.actor_critic.best_action(self.session, observation, goals=goals, deterministic=deterministic)
        return actions[0]

    def _train_actor(self, buffer, env_steps):
        # train actor for multiple epochs on all collected experience
        summary = None
        actor_step = self.actor_step.eval(session=self.session)

        kl_running_avg = 0.0
        early_stop = False

        for epoch in range(self.params.ppo_epochs):
            buffer.shuffle()

            for i in range(0, len(buffer), self.params.batch_size):
                with_summaries = self._should_write_summaries(actor_step) and summary is None
                summaries = [self.actor_summaries] if with_summaries else []

                start, end = i, i + self.params.batch_size
                feed_dict = self.input_dict(buffer, start, end)

                result = self.session.run(
                    [self.objectives.sample_kl, self.train_actor] + summaries,
                    feed_dict=feed_dict)

                actor_step += 1
                self._maybe_save(actor_step, env_steps)

                if with_summaries:
                    summary = result[-1]
                    self.summary_writer.add_summary(summary, global_step=env_steps)

                sample_kl = result[0]
                kl_running_avg = (kl_running_avg + sample_kl) / 2  # running avg with exponential weights for past

                if kl_running_avg > self.params.target_kl:
                    log.info(
                        'Early stopping after %d/%d epochs because of high KL divergence %f > %f',
                        epoch + 1, self.params.ppo_epochs, sample_kl, self.params.target_kl,
                    )
                    early_stop = True
                    break

            if early_stop:
                log.info('Early stopping after %d of %d epochs...', epoch + 1, self.params.ppo_epochs)
                break

        return actor_step

    def _train_critic(self, buffer, env_steps):
        # train critic
        summary = None
        critic_step = self.critic_step.eval(session=self.session)

        prev_loss = 1e10
        for epoch in range(self.params.ppo_epochs):
            losses = []
            buffer.shuffle()

            for i in range(0, len(buffer), self.params.batch_size):
                with_summaries = self._should_write_summaries(critic_step) and summary is None
                summaries = [self.critic_summaries] if with_summaries else []

                start, end = i, i + self.params.batch_size
                feed_dict = self.input_dict(buffer, start, end)

                result = self.session.run(
                    [self.objectives.critic_loss, self.train_critic] + summaries,
                    feed_dict=feed_dict)

                critic_step += 1
                losses.append(result[0])

                if with_summaries:
                    summary = result[-1]
                    self.summary_writer.add_summary(summary, global_step=env_steps)

            # check loss improvement at the end of each epoch, early stop if necessary
            avg_loss = np.mean(losses)
            if avg_loss >= prev_loss:
                log.info('Early stopping after %d epochs because critic did not improve', epoch)
                log.info('Was %.4f now %.4f, ratio %.3f', prev_loss, avg_loss, avg_loss / prev_loss)
                break
            prev_loss = avg_loss

    def _train(self, buffer, env_steps):
        step = self._train_actor(buffer, env_steps)
        self._train_critic(buffer, env_steps)
        return step

    def _learn_loop(self, multi_env):
        """Main training loop."""
        step, env_steps = self.session.run([self.actor_step, self.total_env_steps])

        env_obs = multi_env.reset()
        observations, goals = main_observation(env_obs), goal_observation(env_obs)
        buffer = PPOBuffer()

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
                    actions, action_probs, values = self.actor_critic.invoke(self.session, observations, goals=goals)

                    # wait for all the workers to complete an environment step
                    env_obs, rewards, dones, infos = multi_env.step(actions)
                    new_observations, new_goals = main_observation(env_obs), goal_observation(env_obs)

                    # add experience from all environments to the current buffer
                    buffer.add(observations, actions, action_probs, rewards, dones, values, goals)
                    observations = new_observations
                    goals = new_goals

                    num_steps += num_env_steps(infos)

                # last step values are required for TD-return calculation
                _, _, values = self.actor_critic.invoke(self.session, observations, goals=goals)
                buffer.values.append(values)

            env_steps += num_steps

            # calculate discounted returns and GAE
            buffer.finalize_batch(self.params.gamma, self.params.gae_lambda)

            # update actor and critic
            with timing.timeit('train'):
                step = self._train(buffer, env_steps)

            avg_reward = multi_env.calc_avg_rewards(n=self.params.stats_episodes)
            avg_length = multi_env.calc_avg_episode_lengths(n=self.params.stats_episodes)
            fps = num_steps / (time.time() - batch_start)

            self._maybe_print(step, env_steps, avg_reward, avg_length, fps, timing)
            self._maybe_aux_summaries(env_steps, avg_reward, avg_length)
            self._maybe_update_avg_reward(avg_reward, multi_env.stats_num_episodes())

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
