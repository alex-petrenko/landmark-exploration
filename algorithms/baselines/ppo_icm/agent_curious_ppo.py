import copy
import time
from functools import partial

import tensorflow as tf

from algorithms.algo_utils import num_env_steps, extract_keys, maybe_extract_key
from algorithms.baselines.ppo.agent_ppo import AgentPPO, PPOBuffer
from algorithms.curiosity.curiosity_model import CuriosityModel
from algorithms.env_wrappers import get_observation_space
from algorithms.tf_utils import placeholder_from_space, merge_summaries
from utils.timing import Timing
from utils.utils import AttrDict


class CuriousPPOBuffer(PPOBuffer):
    def __init__(self):
        super(CuriousPPOBuffer, self).__init__()
        self.next_obs = None

    def reset(self):
        super(CuriousPPOBuffer, self).reset()
        self.next_obs = []

    # noinspection PyMethodOverriding
    def add(self, obs, next_obs, actions, action_probs, rewards, dones, values):
        """Append one-step data to the current batch of observations."""
        args = copy.copy(locals())
        super(CuriousPPOBuffer, self)._add_args(args)


class AgentCuriousPPO(AgentPPO):
    """PPO with a curiosity module (ICM or RND)"""
    class Params(AgentPPO.Params):
        """Hyperparams for curious PPO"""
        def __init__(self, experiment_name):
            super(AgentCuriousPPO.Params, self).__init__(experiment_name)

            # add more params here  #TODO: Aleksei give defaults
            self.cm_beta = 1
            self.cm_lr_scale = 1e-3
            self.clip_bonus = 1
            self.stack_past_frames = 3
            self.forward_fc = 512
            self.prediction_bonus_coeff = 0.05  # scaling factor for prediction bonus vs env rewards

        @staticmethod
        def filename_prefix():
            return 'curious_ppo_'

    def __init__(self, make_env_func, params):
        super(AgentCuriousPPO, self).__init__(make_env_func, params)

        env = self.make_env_func()  # we need it to query observation shape, number of actions, etc.
        self.ph_next_observations = placeholder_from_space(get_observation_space(env))
        self.prediction_curiosity_bonus = None  # updated in self.add_cm_objectives()

        # Create graph for curiosity module (ICM)
        self._cm = CuriosityModel(
            env, self.ph_observations, self.ph_next_observations, self.ph_actions, params.stack_past_frames,
            params.forward_fc, params=params)

        # add ICM loss keys to objective function
        self.objectives.update(self.add_cm_objectives())  # TODO: will this merging overwrite some keys?

        self.add_curiosity_summaries()

        self.cm_summaries = merge_summaries(collections=['cm'])
        self.cm_step = tf.Variable(0, trainable=False, dtype=tf.int64, name='cm_step')

        cm_opt = tf.train.AdamOptimizer(learning_rate=self.params.learning_rate, name='cm_opt')
        self.train_cm = cm_opt.minimize(self.objectives.model_loss, global_step=self.cm_step)

    def add_cm_objectives(self):
        # model losses
        forward_loss_batch = 0.5 * tf.square(self._cm.encoded_next_obs - self._cm.predicted_obs)
        forward_loss_batch = tf.reduce_mean(forward_loss_batch, axis=1) * self._cm.feature_vector_size
        forward_loss = tf.reduce_mean(forward_loss_batch)

        bonus = self.params.prediction_bonus_coeff * forward_loss_batch
        self.prediction_curiosity_bonus = tf.clip_by_value(bonus, -self.params.clip_bonus, self.params.clip_bonus)

        inverse_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self._cm.predicted_actions, labels=self.ph_actions,
        ))

        cm_beta = self.params.cm_beta
        model_loss = forward_loss * cm_beta + inverse_loss * (1.0 - cm_beta)
        model_loss = self.params.cm_lr_scale * model_loss
        return AttrDict(locals())

    def add_curiosity_summaries(self):
        obj = self.objectives
        with tf.name_scope('losses'):
            cm_scalar = partial(tf.summary.scalar, collections=['cm'])
            cm_scalar('curiosity_forward_loss', obj.forward_loss)
            cm_scalar('curiosity_inverse_loss', obj.inverse_loss)
            cm_scalar('curiosity_module_loss', obj.model_loss)

    def _train(self, buffer, env_steps):
        step = self._train_actor(buffer, env_steps)
        self._train_critic(buffer, env_steps)
        self._train_curiosity_module(buffer, env_steps)
        return step

    def _prediction_curiosity_bonus(self, observations, actions, next_obs):
        bonuses = self.session.run(
            self.prediction_curiosity_bonus,
            feed_dict={
                self.ph_actions: actions,
                self.ph_observations: observations,
                self.ph_next_observations: next_obs,
            }
        )
        return bonuses

    def _train_curiosity_module(self, buffer, env_steps):
        """
        Actually do a single iteration of training. See the computational graph in the ctor to figure out
        the details.
        """
        observations = buffer.obs
        actions = buffer.actions
        next_obs = buffer.next_obs
        step = self.cm_step.eval(session=self.session)
        summary = None

        for i in range(0, len(buffer), self.params.batch_size):
            with_summaries = self._should_write_summaries(step) and summary is None
            summaries = [self.cm_summaries] if with_summaries else []

            start, end = i, i + self.params.batch_size

            result = self.session.run(
                [self.train_cm] + summaries,
                feed_dict={
                    self.ph_observations: observations[start:end],
                    self.ph_next_observations: next_obs[start:end],
                    self.ph_actions: actions[start:end],
                },
            )

            if with_summaries:
                summary = result[1]
                self.summary_writer.add_summary(summary, global_step=env_steps)

    def _learn_loop(self, multi_env):
        """Main training loop."""
        # env_steps used in tensorboard (and thus, our results)
        # actor_step used as global step for training
        step, env_steps = self.session.run([self.actor_step, self.total_env_steps])

        observations = multi_env.reset()
        img_obs = maybe_extract_key(observations, 'obs')
        buffer = CuriousPPOBuffer()

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
                    actions, action_probs, values = self.actor_critic.invoke(self.session, observations)

                    # wait for all the workers to complete an environment step
                    new_observation, rewards, dones, infos = multi_env.step(actions)

                    # calculate curiosity bonus
                    next_img_obs = maybe_extract_key(new_observation, 'obs')  # TODO: use this for goal and current_obs
                    bonuses = self._prediction_curiosity_bonus(img_obs, actions, next_img_obs)
                    rewards += bonuses

                    # add experience from environment to the current buffer
                    buffer.add(observations, new_observation, actions, action_probs, rewards, dones, values)
                    observations = new_observation
                    img_obs = next_img_obs

                    num_steps += num_env_steps(infos)

                # last step values are required for TD-return calculation
                _, _, values = self.actor_critic.invoke(self.session, observations)
                buffer.values.append(values)

            env_steps += num_steps

            # calculate discounted returns and GAE
            buffer.finalize_batch(self.params.gamma, self.params.gae_lambda)

            # update actor and critic and CM
            with timing.timeit('train'):
                step = self._train(buffer, env_steps)

            avg_reward = multi_env.calc_avg_rewards(n=self.params.stats_episodes)
            avg_length = multi_env.calc_avg_episode_lengths(n=self.params.stats_episodes)
            fps = num_steps / (time.time() - batch_start)

            self._maybe_print(step, avg_reward, avg_length, fps, timing)
            self._maybe_aux_summaries(env_steps, avg_reward, avg_length)
            self._maybe_update_avg_reward(avg_reward, multi_env.stats_num_episodes())
