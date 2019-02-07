"""Utilities."""

import logging
import operator
import os
from os.path import join

import numpy as np
import psutil
from colorlog import ColoredFormatter

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

formatter = ColoredFormatter(
    "%(log_color)s[%(asctime)s] %(message)s",
    datefmt=None,
    reset=True,
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'white,bold',
        'INFOV': 'cyan,bold',
        'WARNING': 'yellow',
        'ERROR': 'red,bold',
        'CRITICAL': 'red,bg_white',
    },
    secondary_log_colors={},
    style='%'
)
ch.setFormatter(formatter)

log = logging.getLogger('rl')
log.setLevel(logging.DEBUG)
log.handlers = []  # No duplicated handlers
log.propagate = False  # workaround for duplicated logs in ipython
log.addHandler(ch)


# general utilities

class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    def __init__(self, d=None):
        if d is not None:
            for key, value in d.items():
                self[key] = value


def op_with_idx(x, op):
    assert len(x) > 0

    best_idx = 0
    best_x = x[best_idx]
    for i, item in enumerate(x):
        if op(item, best_x):
            best_x = item
            best_idx = i

    return best_x, best_idx


def min_with_idx(x):
    return op_with_idx(x, operator.lt)


def max_with_idx(x):
    return op_with_idx(x, operator.gt)


# numpy stuff

def numpy_all_the_way(list_of_arrays):
    """Turn a list of numpy arrays into a 2D numpy array."""
    shape = list(list_of_arrays[0].shape)
    shape[:0] = [len(list_of_arrays)]
    arr = np.concatenate(list_of_arrays).reshape(shape)
    return arr


def numpy_flatten(list_of_arrays):
    """Turn a list of numpy arrays into a 1D numpy array (flattened)."""
    return np.concatenate(list_of_arrays, axis=0)


# os-related stuff

def memory_consumption_mb():
    """Memory consumption of the current process."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


# working with filesystem

def ensure_dir_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def remove_if_exists(file):
    if os.path.isfile(file):
        os.remove(file)


def project_root():
    """
    Keep models, parameters and summaries at the root of this project's directory tree.
    :return: full path to the root dir of this project.
    """
    return os.path.dirname(os.path.dirname(__file__))


def experiments_dir():
    return ensure_dir_exists(join(project_root(), 'train_dir'))


def experiment_dir(experiment, experiments_root=None):
    if experiments_root is None:
        experiments_root = experiments_dir()
    else:
        experiments_root = join(experiments_dir(), experiments_root)

    return ensure_dir_exists(join(experiments_root, experiment))


def model_dir(experiment_dir_):
    return ensure_dir_exists(join(experiment_dir_, '.model'))


def summaries_dir(experiment_dir_):
    return ensure_dir_exists(join(experiment_dir_, '.summary'))


def data_dir(experiment_dir_):
    return ensure_dir_exists(join(experiment_dir_, '.data'))


def vis_dir(experiment_dir_):
    return ensure_dir_exists(join(experiment_dir_, '.vis'))


def get_experiment_name(env_id, name):
    return '{}-{}'.format(env_id, name)
