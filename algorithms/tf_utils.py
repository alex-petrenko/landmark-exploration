import tensorflow as tf


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


# handy placeholder utils, courtesy of https://github.com/openai/spinningup

def combined_shape(length, shape=None):
    if shape is None:
        return length,
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None, dim))


def placeholders(*args):
    return [placeholder(dim) for dim in args]


def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError


def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]


# summaries

def observation_summaries(ph_observations, collections=None):
    if len(ph_observations.shape) >= 4:
        # first three channels
        tf.summary.image('observations', ph_observations[:, :, :, :3], collections=collections)

        # output also last channel if we have more channels than we can display
        if ph_observations.shape[-1].value > 4:
            tf.summary.image('observations_last_channel', ph_observations[:, :, :, -1:], collections=collections)


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
