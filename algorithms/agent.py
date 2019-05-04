"""
Base classes for RL agent implementations with some boilerplate.

"""
import gc
import time
from collections import deque

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim

from utils.decay import LinearDecay
from utils.gifs import encode_gif
from utils.params import Params
from utils.plot import HEATMAP_FIGURE_ID
from utils.tensorboard import visualize_matplotlib_figure_tensorboard
from utils.utils import log, model_dir, summaries_dir, memory_consumption_mb, numpy_all_the_way


class TrainStatus:
    SUCCESS, FAILURE = range(2)


# noinspection PyMethodMayBeStatic,PyUnusedLocal
class Agent:
    def __init__(self, params):
        self.params = params

    def initialize(self):
        pass

    def finalize(self):
        pass

    def analyze_observation(self, observation):
        """Default implementation, may be or may not be overridden."""
        return None

    def best_action(self, observation):
        """Must be overridden in derived classes."""
        raise NotImplementedError('Subclasses should implement {}'.format(self.best_action.__name__))


class AgentRandom(Agent):
    class Params(Params):
        def __init__(self, experiment_name):
            super(AgentRandom.Params, self).__init__(experiment_name)

        @staticmethod
        def filename_prefix():
            return 'random_'

    def __init__(self, make_env_func, params, close_env=True):
        super(AgentRandom, self).__init__(params)
        env = make_env_func()
        self.action_space = env.action_space
        if close_env:
            env.close()

    def best_action(self, *args, **kwargs):
        return self.action_space.sample()


# noinspection PyAbstractClass
class AgentLearner(Agent):
    class AgentParams(Params):
        def __init__(self, experiment_name):
            super(AgentLearner.AgentParams, self).__init__(experiment_name)
            self.use_gpu = True
            self.gpu_mem_fraction = 1.0
            self.initial_save_rate = 2500

            self.stats_episodes = 100  # how many rewards to average to measure performance

            self.gif_save_rate = 150  # number of seconds to wait before saving another gif to tensorboard
            self.gif_summary_num_envs = 2
            self.num_position_histograms = 200  # number of position heatmaps to aggregate
            self.heatmap_save_rate = 60

            self.episode_horizon = -1  # standard environment horizon

            self.use_env_map = False

    def __init__(self, params):
        super(AgentLearner, self).__init__(params)
        self.session = None  # actually created in "initialize" method
        self.saver = None

        tf.reset_default_graph()

        self.summary_rate_decay = LinearDecay([(0, 100), (1000000, 2000), (10000000, 10000)], staircase=100)
        self.save_rate_decay = LinearDecay([(0, self.params.initial_save_rate), (1000000, 5000)], staircase=100)

        self.initial_best_avg_reward = tf.constant(-1e3)
        self.best_avg_reward = tf.Variable(self.initial_best_avg_reward)
        self.total_env_steps = tf.Variable(0, dtype=tf.int64)

        def update_best_value(best_value, new_value):
            return tf.assign(best_value, tf.maximum(new_value, best_value))
        self.avg_reward_placeholder = tf.placeholder(tf.float32, [], 'new_avg_reward')
        self.update_best_reward = update_best_value(self.best_avg_reward, self.avg_reward_placeholder)
        self.total_env_steps_placeholder = tf.placeholder(tf.int64, [], 'new_env_steps')
        self.update_env_steps = tf.assign(self.total_env_steps, self.total_env_steps_placeholder)

        summary_dir = summaries_dir(self.params.experiment_dir())
        self.summary_writer = tf.summary.FileWriter(summary_dir)

        self.position_histograms = deque([], maxlen=self.params.num_position_histograms)

        self._last_trajectory_summary = 0  # timestamp of the latest trajectory summary written
        self._last_coverage_summary = 0  # timestamp of the latest coverage summary written

        self.map_img = self.coord_limits = None

    def initialize(self):
        """Start the session."""
        self.saver = tf.train.Saver(max_to_keep=3)
        all_vars = tf.trainable_variables()
        log.debug('Variables:')
        slim.model_analyzer.analyze_vars(all_vars, print_info=True)

        gpu_options = tf.GPUOptions()
        if self.params.gpu_mem_fraction != 1.0:
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=self.params.gpu_mem_fraction)

        config = tf.ConfigProto(
            device_count={'GPU': 1 if self.params.use_gpu else 0},
            gpu_options=gpu_options,
            log_device_placement=False,
        )
        self.session = tf.Session(config=config)
        self.initialize_variables()
        self.params.serialize()

        log.info('Initialized!')

    def initialize_variables(self):
        checkpoint_dir = model_dir(self.params.experiment_dir())
        try:
            self.saver.restore(self.session, tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir))
        except ValueError:
            log.info('Didn\'t find a valid restore point, start from scratch')
            self.session.run(tf.global_variables_initializer())

    def finalize(self):
        if self.session is not None:
            self.session.close()
        gc.collect()

    def process_infos(self, infos):
        for i, info in enumerate(infos):
            if 'previous_histogram' in info:
                self.position_histograms.append(info['previous_histogram'])

    def _maybe_save(self, step, env_steps):
        self.params.ensure_serialized()
        save_every = self.save_rate_decay.at(step)
        if (step + 1) % save_every == 0:
            self._save(step, env_steps)

    def _save(self, step, env_steps):
        log.info('Training step #%d, env steps: %.1fM, saving...', step, env_steps / 1000000)
        saver_path = model_dir(self.params.experiment_dir()) + '/' + self.__class__.__name__
        self.session.run(self.update_env_steps, feed_dict={self.total_env_steps_placeholder: env_steps})
        self.saver.save(self.session, saver_path, global_step=step)

    def _should_write_summaries(self, step):
        summaries_every = self.summary_rate_decay.at(step)
        return (step + 1) % summaries_every == 0

    def _maybe_update_avg_reward(self, avg_reward, stats_num_episodes):
        if stats_num_episodes > self.params.stats_episodes:
            curr_best_reward = self.best_avg_reward.eval(session=self.session)
            if avg_reward > curr_best_reward + 1e-6:
                log.warn('New best reward %.6f (was %.6f)!', avg_reward, curr_best_reward)
                self.session.run(self.update_best_reward, feed_dict={self.avg_reward_placeholder: avg_reward})

    def _report_basic_summaries(self, fps, env_steps):
        summary = tf.Summary()
        summary.value.add(tag='0_aux/fps', simple_value=float(fps))

        memory_mb = memory_consumption_mb()
        summary.value.add(tag='0_aux/master_process_memory_mb', simple_value=float(memory_mb))

        self.summary_writer.add_summary(summary, env_steps)
        self.summary_writer.flush()

    def _maybe_trajectory_summaries(self, trajectory_buffer, env_steps):
        time_since_last = time.time() - self._last_trajectory_summary
        if time_since_last < self.params.gif_save_rate or not trajectory_buffer.complete_trajectories:
            return

        start_gif_summaries = time.time()

        self._last_trajectory_summary = time.time()
        num_envs = self.params.gif_summary_num_envs

        trajectories = [
            numpy_all_the_way(t.obs)[:, :, :, -3:] for t in trajectory_buffer.complete_trajectories[:num_envs]
        ]
        self._write_gif_summaries(tag='obs_trajectories', gif_images=trajectories, step=env_steps)
        log.info('Took %.3f seconds to write gif summaries', time.time() - start_gif_summaries)

    def _write_gif_summaries(self, tag, gif_images, step, fps=12):
        """Logs list of input image vectors (nx[time x w h x c]) into GIFs."""
        def gen_gif_summary(img_stack_):
            img_list = np.split(img_stack_, img_stack_.shape[0], axis=0)
            enc_gif = encode_gif([i[0] for i in img_list], fps=fps)
            thwc = img_stack_.shape
            im_summ = tf.Summary.Image()
            im_summ.height = thwc[1]
            im_summ.width = thwc[2]
            im_summ.colorspace = 1  # greyscale (RGB=3, RGBA=4)
            im_summ.encoded_image_string = enc_gif
            return im_summ

        gif_summaries = []
        for nr, img_stack in enumerate(gif_images):
            gif_summ = gen_gif_summary(img_stack)
            gif_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr), image=gif_summ))

        summary = tf.Summary(value=gif_summaries)
        self.summary_writer.add_summary(summary, step)

    def _maybe_coverage_summaries(self, env_steps):
        time_since_last = time.time() - self._last_coverage_summary
        if time_since_last < self.params.heatmap_save_rate:
            return
        if len(self.position_histograms) == 0:
            return

        self._last_coverage_summary = time.time()
        self._write_position_heatmap_summaries(tag='position_coverage', step=env_steps)

    def _write_position_heatmap_summaries(self, tag, step):
        summed_histogram = np.zeros_like(self.position_histograms[0])
        for hist in self.position_histograms:
            summed_histogram += hist
        summed_histogram += 1  # min shouldn't be 0 (for log scale)

        fig = plt.figure(num=HEATMAP_FIGURE_ID, figsize=(4, 4))
        fig.clear()
        plt.imshow(
            summed_histogram.T,
            norm=colors.LogNorm(vmin=summed_histogram.min(), vmax=summed_histogram.max()),
            cmap='RdBu_r',
        )
        plt.gca().invert_yaxis()
        plt.colorbar()

        summary = visualize_matplotlib_figure_tensorboard(fig, tag)
        self.summary_writer.add_summary(summary, step)
