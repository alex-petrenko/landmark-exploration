import math

import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete

from utils.utils import log


# general tensorflow utils

def dense(x, layer_size, regularizer=None, activation=tf.nn.relu):
    return tf.contrib.layers.fully_connected(
        x,
        layer_size,
        activation_fn=activation,
        weights_regularizer=regularizer,
        biases_regularizer=regularizer,
    )


def conv(x, num_filters, kernel_size, stride=1, regularizer=None, scope=None):
    return tf.contrib.layers.conv2d(
        x,
        num_filters,
        kernel_size,
        stride=stride,
        weights_regularizer=regularizer,
        biases_regularizer=regularizer,
        scope=scope,
    )


def conv_t(x, num_filters, kernel_size, strides=1, padding='SAME', regularizer=None, activation=tf.nn.relu):
    return tf.layers.conv2d_transpose(
        x,
        num_filters,
        kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        kernel_regularizer=regularizer,
        bias_regularizer=regularizer,
    )


def flatten(x):
    return tf.contrib.layers.flatten(x)


def count_total_parameters():
    """
    Returns total number of trainable parameters in the current tf graph.
    https://stackoverflow.com/a/38161314/1645784
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        total_parameters += variable_parameters
    return total_parameters


def tf_shape(tensor):
    return tensor.get_shape().as_list()


# handy placeholder utils, courtesy of https://github.com/openai/spinningup

def combined_shape(length, shape=None):
    if shape is None:
        return length,
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def placeholder(dim=None, dtype=tf.float32):
    return tf.placeholder(dtype=dtype, shape=combined_shape(None, dim))


def placeholders(*args, dtype=tf.float32):
    return [placeholder(dim, dtype) for dim in args]


def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape, space.dtype)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError


def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]


# summaries

def image_summaries_rgb(imgs, name='observations', collections=None):
    if len(imgs.shape) >= 4:  # [batch, w, h, channels]
        # first three channels
        tf.summary.image(name, imgs[:, :, :, -3:], collections=collections)


def summary_avg_min_max(name, x, collections=None):
    tf.summary.scalar(f'{name}_avg', tf.reduce_mean(x), collections)
    tf.summary.scalar(f'{name}_max', tf.reduce_max(x), collections)
    tf.summary.scalar(f'{name}_min', tf.reduce_min(x), collections)


def merge_summaries(collections=None, scopes=None):
    if collections is None:
        collections = []
    if scopes is None:
        scopes = []

    summaries = [tf.summary.merge_all(key=c) for c in collections]
    summaries += [tf.summary.merge_all(scope=s) for s in scopes]
    return tf.summary.merge(summaries)


def put_kernels_on_grid(kernel, pad=1):
    """
    Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.

    Courtesy of: https://gist.github.com/kukuruza/03731dc494603ceab0c5

    :param kernel: tensor of shape [Y, X, NumChannels, NumKernels]
    :param pad: number of black pixels around each filter (between them)
    :return: Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    """

    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(math.sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1:
                    log.warning('Who would enter a prime number of filters?')
                return i, int(n / i)

    (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')

    # X and Y dimensions, w.r.t. padding
    height = kernel.get_shape()[0] + 2 * pad
    width = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2))
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, height * grid_Y, width, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, width * grid_X, height * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x
