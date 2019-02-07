import copy
import math
import random
import time
from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from algorithms.agent import AgentLearner, TrainStatus
from algorithms.algo_utils import calculate_gae, EPS, num_env_steps
from algorithms.encoders import make_encoder
from algorithms.env_wrappers import get_observation_space
from algorithms.models import make_model
from algorithms.multi_env import MultiEnv
from algorithms.tf_utils import dense, count_total_parameters, placeholder_from_space, placeholders, \
    observation_summaries, summary_avg_min_max, merge_summaries, tf_shape
from algorithms.tmax.graph_encoders import make_graph_encoder
from algorithms.tmax.reachability import ReachabilityNetwork
from algorithms.tmax.topological_map import TopologicalMap
from utils.distributions import CategoricalProbabilityDistribution
from utils.utils import log, AttrDict, max_with_idx


def encode_obs_and_neighbors(env, observations, neighbors, num_neighbors, reg, params):
    """
    As an input we're given (current_observation, current_neighborhood), where current_observation is
    just an env. observation, and current_neighborhood is a set of env. observations topologically close to the
    current one (0 or 1 edges away in the topological graph).
    """

    obs_encoder = tf.make_template(
        'obs_enc', make_encoder, create_scope_now_=True, env=env, regularizer=reg, params=params,
    )
    current_obs_encoded = obs_encoder(observations).encoded_input

    if params.use_neighborhood_encoder:
        obs_shape = tf_shape(observations)[1:]
        neighbors = tf.reshape(neighbors, [-1] + obs_shape)  # turn into a single big batch to encode all at once
        neighbors = obs_encoder(neighbors).encoded_input
        neighborhood_encoder = make_graph_encoder(neighbors, num_neighbors, params, 'graph_enc')

        encoded_input = tf.concat([current_obs_encoded, neighborhood_encoder.encoded_neighborhoods], axis=1)
    else:
        encoded_input = current_obs_encoded

    return encoded_input


class ActorCritic:
    def __init__(self, env, ph_observations, ph_neighbors, ph_num_neighbors, params):
        self.ph_observations = ph_observations
        self.ph_neighbors = ph_neighbors
        self.ph_num_neighbors = ph_num_neighbors

        num_actions = env.action_space.n

        reg = None  # don't use L2 regularization

        # actor computation graph
        actor_encoder = tf.make_template('act_enc', encode_obs_and_neighbors, create_scope_now_=True)
        actor_encoded_obs = actor_encoder(env, ph_observations, ph_neighbors, ph_num_neighbors, reg, params)
        actor_model = make_model(actor_encoded_obs, reg, params, 'act_mdl')

        actions_fc = dense(actor_model.latent, params.model_fc_size // 2, reg)
        action_logits = tf.contrib.layers.fully_connected(actions_fc, num_actions, activation_fn=None)
        self.best_action_deterministic = tf.argmax(action_logits, axis=1)
        self.actions_distribution = CategoricalProbabilityDistribution(action_logits)
        self.act = self.actions_distribution.sample()
        self.action_prob = self.actions_distribution.probability(self.act)

        # critic computation graph
        value_encoder = tf.make_template('val_enc', encode_obs_and_neighbors, create_scope_now_=True)
        value_encoded_obs = value_encoder(env, ph_observations, ph_neighbors, ph_num_neighbors, reg, params)
        value_model = make_model(value_encoded_obs, reg, params, 'val_mdl')

        value_fc = dense(value_model.latent, params.model_fc_size // 2, reg)
        self.value = tf.squeeze(tf.contrib.layers.fully_connected(value_fc, 1, activation_fn=None), axis=[1])

        log.info('Total parameters in the model: %d', count_total_parameters())

    def input_dict(self, observations, neighbors, num_neighbors):
        feed_dict = {self.ph_observations: observations}
        if self.ph_neighbors is not None and self.ph_num_neighbors is not None:
            feed_dict[self.ph_neighbors] = neighbors
            feed_dict[self.ph_num_neighbors] = num_neighbors
        return feed_dict

    def invoke(self, session, observations, neighbors, num_neighbors, deterministic=False):
        ops = [
            self.best_action_deterministic if deterministic else self.act,
            self.action_prob,
            self.value,
        ]
        feed_dict = self.input_dict(observations, neighbors, num_neighbors)
        actions, action_prob, values = session.run(ops, feed_dict=feed_dict)
        return actions, action_prob, values

    def best_action(self, session, observations, neighbors, num_neighbors, deterministic=False):
        feed_dict = self.input_dict(observations, neighbors, num_neighbors)
        actions = session.run(
            self.best_action_deterministic if deterministic else self.act, feed_dict=feed_dict,
        )
        return actions


class TmaxPPOBuffer:
    def __init__(self):
        self.obs = self.actions = self.action_probs = self.rewards = self.dones = self.values = None
        self.advantages = self.returns = None
        self.neighbors, self.num_neighbors = None, None

    def reset(self):
        self.obs, self.actions, self.action_probs, self.rewards, self.dones, self.values = [], [], [], [], [], []
        self.neighbors, self.num_neighbors = [], []
        self.advantages = self.returns = None

    def add(self, obs, actions, action_probs, rewards, dones, values, neighbors, num_neighbors):
        """Append one-step data to the current batch of observations."""
        args = copy.copy(locals())
        s = str(args)
        for arg_name, arg_value in args.items():
            if arg_name in self.__dict__ and arg_value is not None:
                self.__dict__[arg_name].append(arg_value)

    def finalize_batch(self, gamma, gae_lambda):
        # convert everything in the buffer into numpy arrays
        for item, x in self.__dict__.items():
            if x is None:
                continue
            self.__dict__[item] = np.asarray(x)

        # calculate discounted returns and GAE
        self.advantages, self.returns = calculate_gae(self.rewards, self.dones, self.values, gamma, gae_lambda)

        # values vector has one extra last value that we don't need
        self.values = self.values[:-1]
        assert self.values.shape == self.advantages.shape

    def generate_batches(self, batch_size):
        num_transitions = self.obs.shape[0] * self.obs.shape[1]
        if num_transitions % batch_size != 0:
            raise Exception(f'Batch size {batch_size} does not divide experience size {num_transitions}')

        chaos = np.random.permutation(num_transitions)
        num_batches = num_transitions // batch_size

        for item, x in self.__dict__.items():
            if x is None:
                continue

            if x.size == 0 or len(x.shape) < 2:
                # "fake" batch data to simplify the code downstream
                self.__dict__[item] = np.array([None] * num_batches)  # each "batch" will just contain a None value
                continue

            data_shape = x.shape[2:]
            x = x.reshape((num_transitions,) + data_shape)  # collapse [rollout, num_envs] into one dimension
            x = x[chaos]
            x = x.reshape((-1, batch_size) + data_shape)  # split into batches
            self.__dict__[item] = x

        assert self.obs.shape[0] == num_batches
        assert self.rewards.shape[0] == num_batches
        assert self.neighbors.shape[0] == num_batches
        return num_batches


class TrajectoryBuffer:
    """Store trajectories for multiple parallel environments."""

    def __init__(self, num_envs):
        """For now we don't need anything except obs and actions."""
        self.obs = [[] for _ in range(num_envs)]
        self.actions = [[] for _ in range(num_envs)]

        self.complete_trajectories = []

    def reset_trajectories(self):
        """Discard old trajectories and start collecting new ones."""
        self.complete_trajectories = []

    def add(self, obs, actions, dones):
        assert len(obs) == len(actions)
        for env_idx in range(len(obs)):
            if dones[env_idx]:
                # finalize the trajectory and put it into a separate buffer
                trajectory = AttrDict({'obs': self.obs[env_idx], 'actions': self.actions[env_idx]})
                self.complete_trajectories.append(trajectory)
                self.obs[env_idx] = []
                self.actions[env_idx] = []
            else:
                self.obs[env_idx].append(obs[env_idx])
                self.actions[env_idx].append(actions[env_idx])


class ReachabilityBuffer:
    """Training data for the reachability network (observation pairs and labels)."""

    def __init__(self, params):
        self.obs_first, self.obs_second, self.labels = [], [], []
        self.params = params

    def extract_data(self, trajectories, bootstrap_period):
        obs_first, obs_second, labels = [], [], []
        for trajectory in trajectories:
            obs = trajectory.obs
            episode_len = len(obs)

            obs_pairs_fraction = self.params.obs_pairs_per_episode if bootstrap_period else 1.0
            num_obs_pairs = int(obs_pairs_fraction * episode_len)

            reachable_thr = self.params.reachable_threshold
            unreachable_thr = self.params.unreachable_threshold

            try:
                for _ in range(num_obs_pairs):
                    # toss a coin to determine if we want a reachable pair or not
                    reachable = np.random.rand() <= 0.5
                    threshold = reachable_thr if reachable else unreachable_thr

                    # sample first obs in a pair
                    first_idx = np.random.randint(0, episode_len - threshold - 1)

                    # sample second obs
                    if reachable:
                        second_idx = np.random.randint(first_idx, first_idx + reachable_thr)
                    else:
                        second_idx = np.random.randint(first_idx + unreachable_thr, episode_len)

                    obs_first.append(obs[first_idx])
                    obs_second.append(obs[second_idx])
                    labels.append(int(reachable))
            except ValueError:
                # just in case, if some episode is e.g. too short for unreachable pair
                log.exception(f'Value error in Reachability buffer! Episode len {episode_len}')

        if len(obs_first) <= 0:
            return

        if len(self.obs_first) <= 0:
            self.obs_first = np.array(obs_first)
            self.obs_second = np.array(obs_second)
            self.labels = np.array(labels, dtype=np.int32)
        else:
            self.obs_first = np.append(self.obs_first, obs_first, axis=0)
            self.obs_second = np.append(self.obs_second, obs_second, axis=0)
            self.labels = np.append(self.labels, labels, axis=0)

        self._discard_data()

        assert len(self.obs_first) == len(self.obs_second)
        assert len(self.obs_first) == len(self.labels)

    def _discard_data(self):
        """Remove some data if the current buffer is too big."""
        target_size = self.params.reachability_target_buffer_size
        if len(self.obs_first) <= target_size:
            return

        self.shuffle_data()
        self.obs_first = self.obs_first[:target_size]
        self.obs_second = self.obs_second[:target_size]
        self.labels = self.labels[:target_size]

    def has_enough_data(self):
        len_data, min_data = len(self.obs_first), self.params.reachability_target_buffer_size // 2
        if len_data < min_data:
            log.info('Not enough data to train reachability net, %d/%d', len_data, min_data)
            return False
        return True

    def shuffle_data(self):
        if len(self.obs_first) <= 0:
            return

        chaos = np.random.permutation(len(self.obs_first))
        self.obs_first = self.obs_first[chaos]
        self.obs_second = self.obs_second[chaos]
        self.labels = self.labels[chaos]


class AgentTMAX(AgentLearner):
    """Agent based on PPO+TMAX algorithm."""

    class Params(AgentLearner.AgentParams):
        """Hyperparams for the algorithm and the training process."""

        def __init__(self, experiment_name):
            """Default parameter values set in ctor."""
            super(AgentTMAX.Params, self).__init__(experiment_name)

            self.gamma = 0.99  # future reward discount
            self.gae_lambda = 0.8
            self.rollout = 64
            self.num_envs = 192  # number of environments to collect the experience from
            self.num_workers = 16  # number of workers used to run the environments

            # actor-critic (encoders and models)
            self.image_enc_name = 'convnet_doom_small'
            self.model_fc_layers = 1
            self.model_fc_size = 256
            self.model_recurrent = False
            self.rnn_rollout = 16

            # ppo-specific
            self.ppo_clip_ratio = 1.1  # we use clip(x, e, 1/e) instead of clip(x, 1+e, 1-e) in the paper
            self.target_kl = 0.02
            self.batch_size = 512
            self.ppo_epochs = 10

            # components of the loss function
            self.initial_entropy_loss_coeff = 0.1
            self.min_entropy_loss_coeff = 0.002

            # TMAX-specific parameters
            self.use_neighborhood_encoder = False
            self.graph_enc_name = 'deepsets'  # 'rnn', 'deepsets'
            self.max_neighborhood_size = 6  # max number of neighbors that can be fed into policy at every timestep
            self.graph_encoder_rnn_size = 256  # size of GRU layer in RNN neighborhood encoder

            self.obs_pairs_per_episode = 0.25  # e.g. for episode of len 300 we will create 75 training pairs
            self.reachable_threshold = 8  # num. of frames between obs, such that one is reachable from the other
            self.unreachable_threshold = 24  # num. of frames between obs, such that one is unreachable from the other
            self.reachability_target_buffer_size = 25000  # target number of training examples to store
            self.reachability_train_epochs = 1
            self.reachability_batch_size = 128

            self.new_landmark_reachability = 0.15  # condition for considering current observation a "new landmark"
            self.loop_closure_reachability = 0.5  # condition for graph loop closure (finding new edge)
            self.map_expansion_reward = 0.05  # reward for finding new vertex or new edge in the topological map

            self.bootstrap_env_steps = 750 * 1000

            # training process
            self.learning_rate = 1e-4
            self.train_for_steps = self.train_for_env_steps = 10 * 1000 * 1000 * 1000
            self.use_gpu = True
            self.initial_save_rate = 1000

        @staticmethod
        def filename_prefix():
            return 'tmax_'

    def __init__(self, make_env_func, params):
        """Initialize PPO computation graph and some auxiliary tensors."""
        super(AgentTMAX, self).__init__(params)

        # separate global_steps
        self.actor_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='actor_step')
        self.critic_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='critic_step')
        self.reach_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='reach_step')

        self.make_env_func = make_env_func
        env = make_env_func()  # we need the env to query observation shape, number of actions, etc.

        self.obs_shape = list(get_observation_space(env).shape)
        self.ph_observations = placeholder_from_space(get_observation_space(env))
        self.ph_actions = placeholder_from_space(env.action_space)  # actions sampled from the policy
        self.ph_advantages, self.ph_returns, self.ph_old_action_probs = placeholders(None, None, None)

        # placeholders for the graph neighborhood
        self.ph_neighbors, self.ph_num_neighbors = None, None
        if params.use_neighborhood_encoder:
            self.ph_neighbors = tf.placeholder(tf.float32, shape=[None, params.max_neighborhood_size] + self.obs_shape)
            self.ph_num_neighbors = tf.placeholder(tf.int32, shape=[None])

        self.actor_critic = ActorCritic(
            env, self.ph_observations, self.ph_neighbors, self.ph_num_neighbors, self.params,
        )

        self.reachability = ReachabilityNetwork(env, params)

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

        reach_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='reach_opt')
        self.train_reachability = reach_opt.minimize(self.reachability.loss, global_step=self.reach_step)

        # summaries
        self.add_summaries()

        self.actor_summaries = merge_summaries(collections=['actor'])
        self.critic_summaries = merge_summaries(collections=['critic'])
        self.reach_summaries = merge_summaries(collections=['reachability'])

        self.saver = tf.train.Saver(max_to_keep=3)

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

        with tf.name_scope('reachability'):
            reachability_scalar = partial(tf.summary.scalar, collections=['reachability'])
            reachability_scalar('reach_loss', self.reachability.loss)

    def _maybe_print(self, step, env_step, avg_rewards, avg_length, fps, t):
        log.info('<====== Step %d, env step %.1fM ======>', step, env_step / 1000000)
        log.info('Avg FPS: %.1f', fps)
        log.info('Experience for batch took %.3f sec (%.1f batches/s)', t.experience, 1.0 / t.experience)
        log.info('Train step for batch took %.3f sec (%.1f batches/s)', t.train, 1.0 / t.train)
        log.info('Train reachability took %.3f sec (%.1f batches/s)', t.reach, 1.0 / t.reach)

        if math.isnan(avg_rewards) or math.isnan(avg_length):
            return

        log.info('Avg. %d episode lenght: %.3f', self.params.stats_episodes, avg_length)
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
        initial_best, best_reward = self.session.run([self.initial_best_avg_reward, self.best_avg_reward])
        if best_reward != initial_best:
            summary.value.add(tag='0_aux/best_reward_ever', simple_value=float(best_reward))

        self.summary_writer.add_summary(summary, env_steps)
        self.summary_writer.flush()

    def _maybe_map_summaries(self, maps, env_steps):
        num_landmarks = [len(m.landmarks) for m in maps]
        num_neighbors = [len(m.neighbor_indices()) for m in maps]
        num_edges = [m.num_undirected_edges() for m in maps]

        avg_num_landmarks = sum(num_landmarks) / len(num_landmarks)
        avg_num_neighbors = sum(num_neighbors) / len(num_neighbors)
        avg_num_edges = sum(num_edges) / len(num_edges)

        summary_obj = tf.Summary()

        def summary(tag, value):
            summary_obj.value.add(tag=f'map/{tag}', simple_value=float(value))

        summary('avg_landmarks', avg_num_landmarks)
        summary('max_landmarks', max(num_landmarks))
        summary('avg_neighbors', avg_num_neighbors)
        summary('max_neighbors', max(num_neighbors))
        summary('avg_edges', avg_num_edges)
        summary('max_edges', max(num_edges))

        self.summary_writer.add_summary(summary_obj, env_steps)
        self.summary_writer.flush()

    def best_action(self, observation):
        raise NotImplementedError('Not supported! Use best_action_tmax')

    def best_action_tmax(self, observations, neighbors, num_neighbors, deterministic=False):
        actions = self.actor_critic.best_action(self.session, observations, neighbors, num_neighbors, deterministic)
        return actions[0]

    def _train_actor(self, buffer, env_steps):
        # train actor for multiple epochs on all collected experience
        summary = None
        actor_step = self.actor_step.eval(session=self.session)

        kl_running_avg = 0.0
        early_stop = False
        for epoch in range(self.params.ppo_epochs):
            num_batches = buffer.generate_batches(self.params.batch_size)
            total_num_steps = self.params.ppo_epochs * num_batches

            for i in range(num_batches):
                with_summaries = self._should_write_summaries(actor_step) and summary is None
                summaries = [self.actor_summaries] if with_summaries else []

                result = self.session.run(
                    [self.objectives.sample_kl, self.train_actor] + summaries,
                    feed_dict={
                        self.ph_actions: buffer.actions[i],
                        self.ph_old_action_probs: buffer.action_probs[i],
                        self.ph_advantages: buffer.advantages[i],
                        self.ph_returns: buffer.returns[i],
                        **self.actor_critic.input_dict(buffer.obs[i], buffer.neighbors[i], buffer.num_neighbors[i]),
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

    def _train_critic(self, buffer, env_steps):
        # train critic
        summary = None
        critic_step = self.critic_step.eval(session=self.session)

        prev_loss = 1e10
        losses = []

        for epoch in range(self.params.ppo_epochs):
            num_batches = buffer.generate_batches(self.params.batch_size)

            for i in range(num_batches):
                with_summaries = self._should_write_summaries(critic_step) and summary is None
                summaries = [self.critic_summaries] if with_summaries else []

                result = self.session.run(
                    [self.objectives.critic_loss, self.train_critic] + summaries,
                    feed_dict={
                        self.ph_returns: buffer.returns[i],
                        **self.actor_critic.input_dict(buffer.obs[i], buffer.neighbors[i], buffer.num_neighbors[i]),
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
                log.info('Was %.4f now %.4f, ratio %.3f', prev_loss, avg_loss, avg_loss / prev_loss)
                break
            prev_loss = avg_loss

    def _train(self, buffer, env_steps):
        step = self._train_actor(buffer, env_steps)
        self._train_critic(buffer, env_steps)
        return step

    def _maybe_train_reachability(self, buffer, env_steps):
        if not buffer.has_enough_data():
            return

        batch_size = self.params.reachability_batch_size
        summary = None
        reach_step = self.reach_step.eval(session=self.session)

        prev_loss = 1e10
        losses = []

        num_epochs = self.params.reachability_train_epochs
        if self._is_bootstrap(env_steps):
            num_epochs = max(10, num_epochs * 2)  # during bootstrap do more epochs to train faster!

        for epoch in range(num_epochs):
            buffer.shuffle_data()
            obs_first, obs_second, labels = buffer.obs_first, buffer.obs_second, buffer.labels

            for i in range(0, len(obs_first) - 1, batch_size):
                with_summaries = self._should_write_summaries(reach_step) and summary is None
                summaries = [self.reach_summaries] if with_summaries else []

                start, end = i, i + batch_size

                result = self.session.run(
                    [self.reachability.loss, self.train_reachability] + summaries,
                    feed_dict={
                        self.reachability.ph_obs_first: obs_first[start:end],
                        self.reachability.ph_obs_second: obs_second[start:end],
                        self.reachability.ph_labels: labels[start:end],
                    }
                )

                reach_step += 1
                losses.append(result[0])

                if with_summaries:
                    summary = result[-1]
                    self.summary_writer.add_summary(summary, global_step=env_steps)

            # check loss improvement at the end of each epoch, early stop if necessary
            avg_loss = np.mean(losses)
            if avg_loss > 0.999 * prev_loss:
                log.info('Early stopping after %d epochs because reachability did not improve enough', epoch)
                log.info('Was %.4f now %.4f, ratio %.3f', prev_loss, avg_loss, avg_loss / prev_loss)
                break
            prev_loss = avg_loss

    def _is_bootstrap(self, env_steps):
        """Check whether we're still in the initial bootstrapping stage."""
        return env_steps < self.params.bootstrap_env_steps

    def update_maps(self, maps, obs, dones, env_steps=None, verbose=False):
        """Omnipotent function for the management of topological maps."""
        assert len(obs) == len(maps)
        num_envs = len(maps)

        bonuses = np.zeros([num_envs])

        if env_steps is None:
            env_steps = self.session.run(self.total_env_steps)

        if self._is_bootstrap(env_steps):
            # don't bother updating the graph when the reachability net isn't trained yet
            return bonuses

        def log_verbose(s, *args):
            if verbose:
                log.debug(s, *args)

        for i, m in enumerate(maps):
            if dones[i]:
                m.reset(obs[i])  # reset graph on episode termination (TODO! there must be a better policy)

        # create a batch of all neighborhood observations from all envs for fast processing on GPU
        neighborhood_obs = []
        current_obs = []
        for i, m in enumerate(maps):
            neighbor_indices = m.neighbor_indices()
            neighborhood_obs.extend([m.landmarks[i] for i in neighbor_indices])
            current_obs.extend([obs[i]] * len(neighbor_indices))

        assert len(neighborhood_obs) == len(current_obs)

        # calculate reachability for all neighborhoods in all envs
        reachabilities = self.reachability.get_reachability(self.session, neighborhood_obs, current_obs)

        new_landmark_candidates = []

        j = 0
        for env_i, m in enumerate(maps):
            neighbor_indices = m.neighbor_indices()
            j_next = j + len(neighbor_indices)
            reachability = reachabilities[j:j_next]

            # optional diagnostic logging
            log_reachability = True
            if verbose and log_reachability:
                neighbor_reachability = {}
                for i, neighbor_idx in enumerate(neighbor_indices):
                    neighbor_reachability[neighbor_idx] = '{:.3f}'.format(reachability[i])
                log_verbose('Env %d reachability: %r', env_i, neighbor_reachability)

            # check if we're far enough from all landmarks in the neighborhood
            max_r = max(reachability)
            if max_r < self.params.new_landmark_reachability:
                # we're far enough from all obs in the neighborhood, might have found something new!
                new_landmark_candidates.append(env_i)
            else:
                # we're still sufficiently close to our neighborhood, but maybe "current landmark" has changed
                max_r, max_r_idx = max_with_idx(reachability)
                m.set_curr_landmark(neighbor_indices[max_r_idx])

            j = j_next

        del neighborhood_obs
        del current_obs

        # Agents in some environments discovered landmarks that are far away from all landmarks in the immediate
        # neighborhood. There are two possibilities:
        # 1) A new landmark should be created and added to the graph
        # 2) We found a "loop closure" - a new edge in a graph

        non_neighborhood_obs = []
        non_neighborhoods = {}
        current_obs = []
        for env_i in new_landmark_candidates:
            m = maps[env_i]
            non_neighbor_indices = m.non_neighbor_indices()
            non_neighborhoods[env_i] = non_neighbor_indices
            non_neighborhood_obs.extend([m.landmarks[i] for i in non_neighbor_indices])
            current_obs.extend([obs[env_i]] * len(non_neighbor_indices))

        # this can be potentially a very large batch, should we divide it into mini-batches?
        assert len(non_neighborhood_obs) == len(current_obs)

        # calculate reachability for all non-neighbors
        reachabilities = []
        if len(non_neighborhood_obs) != 0:
            reachabilities = self.reachability.get_reachability(self.session, non_neighborhood_obs, current_obs)

        j = 0
        for env_i in new_landmark_candidates:
            m = maps[env_i]
            non_neighbor_indices = non_neighborhoods[env_i]
            j_next = j + len(non_neighbor_indices)
            reachability = reachabilities[j:j_next]

            max_r, max_r_idx = -math.inf, -math.inf
            if len(reachability) > 0:
                max_r, max_r_idx = max_with_idx(reachability)

            if max_r > self.params.loop_closure_reachability:
                # current observation is close to some other landmark, "close the loop" by creating a new edge
                m.set_curr_landmark(non_neighbor_indices[max_r_idx])
                log_verbose('Change current landmark to %d (loop closure)', m.curr_landmark_idx)
            else:
                # vertex is relatively far away from all vertex in the graph, we've found a new landmark!
                new_landmark_idx = m.add_landmark(obs[env_i])
                m.set_curr_landmark(new_landmark_idx)

            bonuses[env_i] += self.params.map_expansion_reward  # we found a new vertex or edge! Cool!
            j = j_next

        return bonuses

    def get_neighbors(self, maps, neighbors_buffer):
        if not self.params.use_neighborhood_encoder:
            return None, None

        neighbors_buffer.fill(0)
        num_neighbors = [0] * len(maps)

        for env_idx, m in enumerate(maps):
            n_indices = m.neighbor_indices()
            random.shuffle(n_indices)  # order of neighbors does not matter

            for i, n_idx in enumerate(n_indices):
                if i >= self.params.max_neighborhood_size:
                    log.warning(
                        'Too many neighbors %d, max encoded is %d. Had to ignore some neighbors.',
                        len(n_indices), self.params.max_neighborhood_size,
                    )
                    break

                neighbors_buffer[env_idx, i] = m.landmarks[n_idx]
            num_neighbors[env_idx] = min(len(n_indices), self.params.max_neighborhood_size)

        return neighbors_buffer, num_neighbors

    def _learn_loop(self, multi_env):
        """Main training loop."""
        step, env_steps = self.session.run([self.actor_step, self.total_env_steps])

        observations = multi_env.reset()
        buffer = TmaxPPOBuffer()

        trajectory_buffer = TrajectoryBuffer(multi_env.num_envs)
        reachability_buffer = ReachabilityBuffer(self.params)

        neighbors = np.zeros(
            [multi_env.num_envs, self.params.max_neighborhood_size] + self.obs_shape, dtype=np.uint8,
        )

        maps = [TopologicalMap(obs) for obs in observations]

        def end_of_training(s, es):
            return s >= self.params.train_for_steps or es > self.params.train_for_env_steps

        while not end_of_training(step, env_steps):
            timing = AttrDict({'experience': time.time(), 'rollout': time.time()})
            buffer.reset()

            # collecting experience
            num_steps = 0
            for rollout_step in range(self.params.rollout):
                neighbors, num_neighbors = self.get_neighbors(maps, neighbors)
                actions, action_probs, values = self.actor_critic.invoke(
                    self.session, observations, neighbors, num_neighbors,
                )

                # wait for all the workers to complete an environment step
                new_observations, rewards, dones, infos = multi_env.step(actions)

                bonuses = self.update_maps(maps, new_observations, dones, env_steps)
                rewards += bonuses

                # add experience from all environments to the current buffer(s)
                buffer.add(observations, actions, action_probs, rewards, dones, values, neighbors, num_neighbors)
                trajectory_buffer.add(observations, actions, dones)
                observations = new_observations

                num_steps += num_env_steps(infos, multi_env.num_envs)

            # last step values are required for TD-return calculation
            neighbors, num_neighbors = self.get_neighbors(maps, neighbors)
            _, _, values = self.actor_critic.invoke(self.session, observations, neighbors, num_neighbors)
            buffer.values.append(values)

            timing.experience = time.time() - timing.experience
            env_steps += num_steps

            # calculate discounted returns and GAE
            buffer.finalize_batch(self.params.gamma, self.params.gae_lambda)

            # update actor and critic
            timing.train = time.time()
            if not self._is_bootstrap(env_steps):
                step = self._train(buffer, env_steps)
            timing.train = time.time() - timing.train

            # update reachability net
            timing.reach = time.time()
            reachability_buffer.extract_data(trajectory_buffer.complete_trajectories, self._is_bootstrap(env_steps))
            trajectory_buffer.reset_trajectories()
            self._maybe_train_reachability(reachability_buffer, env_steps)
            timing.reach = time.time() - timing.reach

            avg_reward = multi_env.calc_avg_rewards(n=self.params.stats_episodes)
            avg_length = multi_env.calc_avg_episode_lengths(n=self.params.stats_episodes)
            fps = num_steps / (time.time() - timing.rollout)

            self._maybe_print(step, env_steps, avg_reward, avg_length, fps, timing)
            self._maybe_aux_summaries(env_steps, avg_reward, avg_length, fps)
            self._maybe_map_summaries(maps, env_steps)
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
