import math

import numpy as np
import time
from functools import partial

import tensorflow as tf
from tensorflow.contrib import slim

from algorithms.agent import AgentLearner, summaries_dir
from algorithms.algo_utils import calculate_gae, EPS, extract_key
from algorithms.encoders import make_encoder
from algorithms.models import make_model
from algorithms.multi_env import MultiEnv
from algorithms.tf_utils import dense, count_total_parameters, placeholder_from_space, placeholders, \
    observation_summaries, summary_avg_min_max, merge_summaries
from modules.distributions import CategoricalProbabilityDistribution
from utils.utils import log, AttrDict


class ActorCritic:
    def __init__(self, env, ph_observations, params):
        self.ph_observations = ph_observations

        num_actions = env.action_space.n

        regularizer = None  # don't use L2 regularization

        # actor computation graph
        actor_encoder = make_encoder(env, ph_observations, regularizer, params, 'act_enc')
        actor_model = make_model(actor_encoder.encoded_input, regularizer, params, 'act_mdl')

        actions_fc = dense(actor_model.latent, params.model_fc_size // 2, regularizer)
        action_logits = tf.contrib.layers.fully_connected(actions_fc, num_actions, activation_fn=None)
        self.best_action_deterministic = tf.argmax(action_logits, axis=1)
        self.actions_distribution = CategoricalProbabilityDistribution(action_logits)
        self.act = self.actions_distribution.sample()
        self.action_prob = self.actions_distribution.probability(self.act)

        # critic computation graph
        value_encoder = make_encoder(env, ph_observations, regularizer, params, 'val_enc')
        value_model = make_model(value_encoder.encoded_input, regularizer, params, 'val_mdl')

        value_fc = dense(value_model.latent, params.model_fc_size // 2, regularizer)
        self.value = tf.squeeze(tf.contrib.layers.fully_connected(value_fc, 1, activation_fn=None), axis=[1])

        log.info('Total parameters in the model: %d', count_total_parameters())

    def invoke(self, session, observations, deterministic=False):
        # TODO RECURRENT
        ops = [
            self.best_action_deterministic if deterministic else self.act,
            self.action_prob,
            self.value,
        ]
        actions, action_prob, values = session.run(ops, feed_dict={self.ph_observations: observations})
        return actions, action_prob, values

    def best_action(self, session, observations, deterministic=False):
        actions = session.run(
            self.best_action_deterministic if deterministic else self.act,
            feed_dict={self.ph_observations: observations},
        )
        return actions


class PPOBuffer:
    def __init__(self):
        self.obs = self.actions = self.action_probs = self.rewards = self.dones = self.values = None
        self.advantages = self.returns = None

    def reset(self):
        self.obs, self.actions, self.action_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []
        self.advantages = self.returns = None

    def add(self, obs, actions, action_probs, rewards, dones, values):
        self.obs.append(obs)
        self.actions.append(actions)
        self.action_probs.append(action_probs)
        self.rewards.append(rewards)
        self.dones.append(dones)
        self.values.append(values)

    def finalize_batch(self, gamma, gae_lambda):
        # convert everything in the buffer into numpy arrays
        for item, x in self.__dict__.items():
            if x is None:
                continue
            self.__dict__[item] = np.asarray(x, np.float32)

        # calculate discounted returns and GAE
        self.advantages, self.returns = calculate_gae(self.rewards, self.dones, self.values, gamma, gae_lambda)

        # values vector has one extra last value that we don't need
        self.values = self.values[:-1]
        assert self.values.shape == self.advantages.shape

    def _generate_batches(self, batch_size):
        num_transitions = self.obs.shape[0] * self.obs.shape[1]
        if num_transitions % batch_size != 0:
            raise Exception(f'Batch size {batch_size} does not divide experience size {num_transitions}')

        chaos = np.random.permutation(num_transitions)

        for item, x in self.__dict__.items():
            if x is None or x.size == 0:
                continue

            data_shape = x.shape[2:]
            x = x.reshape((num_transitions,) + data_shape)  # collapse [rollout, num_envs] into one dimension
            x = x[chaos]
            x = x.reshape((-1, batch_size) + data_shape)  # split into batches
            self.__dict__[item] = x

        num_batches = num_transitions // batch_size
        assert self.obs.shape[0] == num_batches
        assert self.rewards.shape[0] == num_batches
        return num_batches

    def _generate_batches_recurrent(self, batch_size, trajectory_len):
        # TODO!
        pass

    def generate_batches(self, batch_size, trajectory_len):
        if trajectory_len <= 1:
            return self._generate_batches(batch_size)
        else:
            return self._generate_batches_recurrent(batch_size, trajectory_len)


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

            self.stack_past_frames = 3
            self.num_input_frames = self.stack_past_frames

            # actor-critic (encoders and models)
            self.image_enc_name = 'convnet_large'
            self.lowdim_enc_name = 'order_invariant'
            self.model_fc_layers = 1
            self.model_fc_size = 256
            self.model_recurrent = False
            self.rnn_rollout = 16

            # ppo-specific
            self.ppo_clip_ratio = 1.1  # we use clip(x, e, 1/e) instead of clip(x, 1+e, 1-e) in the paper
            self.target_kl = 0.02
            self.batch_size = 1024
            self.ppo_epochs = 10

            # components of the loss function
            self.initial_entropy_loss_coeff = 0.1
            self.min_entropy_loss_coeff = 0.002

            # training process
            self.learning_rate = 1e-4
            self.train_for_steps = self.train_for_env_steps = 10 * 1000 * 1000 * 1000
            self.use_gpu = True
            self.initial_save_rate = 1000

        # noinspection PyMethodMayBeStatic
        def filename_prefix(self):
            return 'ppo_'

    def __init__(self, make_env_func, params):
        """Initialize PPO computation graph and some auxiliary tensors."""
        super(AgentPPO, self).__init__(params)

        self.actor_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='actor_step')
        self.critic_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='critic_step')

        self.make_env_func = make_env_func
        env = make_env_func()  # we need the env to query observation shape, number of actions, etc.

        obs_shape = list(env.observation_space.spaces['obs'].shape)
        input_shape = [None] + obs_shape  # add batch dimension
        self.ph_observations = tf.placeholder(tf.float32, shape=input_shape)
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

        self.add_summaries()

        summary_dir = summaries_dir(self.params.experiment_dir())
        self.summary_writer = tf.summary.FileWriter(summary_dir)
        self.actor_summaries = merge_summaries(collections=['actor'])
        self.critic_summaries = merge_summaries(collections=['critic'])

        self.saver = tf.train.Saver(max_to_keep=3)

        log.debug('ppo variables:')
        all_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(all_vars, print_info=True)

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

        # final losses to optimize
        actor_loss = ppo_loss + entropy_loss
        critic_loss = value_loss

        # auxiliary quantities (for tensorboard, logging, early stopping)
        log_p_old = tf.log(old_action_probs + EPS)
        log_p = tf.log(action_probs + EPS)
        sample_kl = tf.reduce_mean(log_p_old - log_p)
        sample_entropy = tf.reduce_mean(-log_p)
        clipped_fraction = tf.reduce_mean(clipped)

        return AttrDict(locals())

    def add_summaries(self):
        obj = self.objectives

        # summaries for the agent and the training process
        with tf.name_scope('obs_summaries'):
            observation_summaries(self.ph_observations, collections=['actor'])

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

    def _maybe_print(self, step, avg_rewards, avg_length, fps, t):
        log.info('<====== Step %d ======>', step)
        log.info('Avg FPS: %.1f', fps)
        log.info('Experience for batch took %.3f sec (%.1f batches/s)', t.experience, 1.0 / t.experience)
        log.info('Train step for batch took %.3f sec (%.1f batches/s)', t.train, 1.0 / t.train)

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

    def best_action(self, observation, deterministic=False):
        observation = extract_key([observation], 'obs')
        actions = self.actor_critic.best_action(self.session, observation, deterministic)
        return actions[0]

    def _train_actor(self, buffer, env_steps, trajectory_length):
        # train actor for multiple epochs on all collected experience
        summary = None
        actor_step = self.actor_step.eval(session=self.session)

        kl_running_avg = 0.0
        early_stop = False
        for epoch in range(self.params.ppo_epochs):
            num_batches = buffer.generate_batches(self.params.batch_size, trajectory_length)
            total_num_steps = self.params.ppo_epochs * num_batches

            for i in range(num_batches):
                with_summaries = self._should_write_summaries(actor_step) and summary is None
                summaries = [self.actor_summaries] if with_summaries else []

                result = self.session.run(
                    [self.objectives.sample_kl, self.train_actor] + summaries,
                    feed_dict={
                        self.ph_observations: buffer.obs[i],
                        self.ph_actions: buffer.actions[i],
                        self.ph_old_action_probs: buffer.action_probs[i],
                        self.ph_advantages: buffer.advantages[i],
                        self.ph_returns: buffer.returns[i],
                    }
                )

                actor_step += 1
                self._maybe_save(actor_step, env_steps)

                if with_summaries:
                    summary = result[-1]
                    self.summary_writer.add_summary(summary, global_step=env_steps)

                sample_kl = result[0]
                kl_running_avg = (kl_running_avg + sample_kl) / 2  # running avg with exponential weights for past

                if kl_running_avg > self.params.target_kl:
                    log.info(
                        'Early stopping after %d/%d steps because of high KL divergence %f > %f',
                        epoch * num_batches + i, total_num_steps, sample_kl, self.params.target_kl,
                    )
                    early_stop = True
                    break

            if early_stop:
                log.info('Early stopping after %d of %d epochs...', epoch, self.params.ppo_epochs)
                break

        return actor_step

    def _train_critic(self, buffer, env_steps, trajectory_length):
        # train critic
        summary = None
        critic_step = self.critic_step.eval(session=self.session)

        prev_loss = 1e10
        losses = []

        for epoch in range(self.params.ppo_epochs):
            num_batches = buffer.generate_batches(self.params.batch_size, trajectory_length)

            for i in range(num_batches):
                with_summaries = self._should_write_summaries(critic_step) and summary is None
                summaries = [self.critic_summaries] if with_summaries else []

                result = self.session.run(
                    [self.objectives.critic_loss, self.train_critic] + summaries,
                    feed_dict={
                        self.ph_observations: buffer.obs[i],
                        self.ph_returns: buffer.returns[i],
                    }
                )

                critic_step += 1
                losses.append(result[0])

                if with_summaries:
                    summary = result[-1]
                    self.summary_writer.add_summary(summary, global_step=env_steps)

            # check loss improvement at the end of each epoch, early stop if necessary
            avg_loss = np.mean(losses)
            if avg_loss > 0.995 * prev_loss:
                log.info('Early stopping after %d epochs because critic did not improve enough', epoch)
                log.info('Was %.3f now %.3f, ratio %.3f', prev_loss, avg_loss, avg_loss / prev_loss)
                break
            prev_loss = avg_loss

    def _train(self, buffer, env_steps):
        trajectory_length = self.params.rnn_rollout if self.params.model_recurrent else 1
        step = self._train_actor(buffer, env_steps, trajectory_length)
        self._train_critic(buffer, env_steps, trajectory_length)
        return step

    def _learn_loop(self, multi_env):
        """Main training loop."""
        step, env_steps = self.session.run([self.actor_step, self.total_env_steps])

        observations = extract_key(multi_env.initial_obs(), 'obs')
        buffer = PPOBuffer()

        def end_of_training(s, es):
            return s >= self.params.train_for_steps or es > self.params.train_for_env_steps

        while not end_of_training(step, env_steps):
            timing = AttrDict({'experience': time.time(), 'rollout': time.time()})
            buffer.reset()

            # collecting experience
            for rollout_step in range(self.params.rollout):
                actions, action_probs, values = self.actor_critic.invoke(self.session, observations)

                # wait for all the workers to complete an environment step
                new_observation, rewards, dones, _ = multi_env.step(actions)
                new_observation = extract_key(new_observation, 'obs')

                # add experience from all environments to the current buffer
                buffer.add(observations, actions, action_probs, rewards, dones, values)
                observations = new_observation

            # last step values are required for TD-return calculation
            _, _, values = self.actor_critic.invoke(self.session, observations)
            buffer.values.append(values)

            timing.experience = time.time() - timing.experience

            # calculate discounted returns and GAE
            num_steps = len(buffer.obs) * multi_env.num_envs
            env_steps += num_steps
            buffer.finalize_batch(self.params.gamma, self.params.gae_lambda)

            # update actor and critic
            timing.train = time.time()
            step = self._train(buffer, env_steps)
            timing.train = time.time() - timing.train

            avg_reward = multi_env.calc_avg_rewards(n=self.params.stats_episodes)
            avg_length = multi_env.calc_avg_episode_lengths(n=self.params.stats_episodes)
            fps = num_steps / (time.time() - timing.rollout)

            self._maybe_print(step, avg_reward, avg_length, fps, timing)
            self._maybe_aux_summaries(env_steps, avg_reward, avg_length)
            self._maybe_update_avg_reward(avg_reward, multi_env.stats_num_episodes())

    def learn(self):
        multi_env = None
        try:
            multi_env = MultiEnv(
                self.params.num_envs,
                self.params.num_workers,
                make_env_func=self.make_env_func,
                stats_episodes=self.params.stats_episodes,
            )

            self._learn_loop(multi_env)
        except Exception as exc:
            log.exception(exc)
        finally:
            log.info('Closing env...')
            if multi_env is not None:
                multi_env.close()
