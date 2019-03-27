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
        super(AttrDict, self).__init__()
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


def ensure_contigious(x):
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)
    return x


# matplotlib

def figure_to_numpy(figure):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param figure a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    figure.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = figure.canvas.get_width_height()
    buffer = np.fromstring(figure.canvas.tostring_argb(), dtype=np.uint8)
    buffer.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buffer = np.roll(buffer, 3, axis=2)
    return buffer


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


def crop_map_image(image_array):
    """Uses Python Image Library (PIL) to crop the borders of an image"""
    # TODO: Use openCV instead of PIL?
    from PIL import Image, ImageChops
    import numpy as np
    buffer = Image.fromarray(image_array)

    def trim(im):
        bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
        diff = ImageChops.difference(im, bg)
        diff = ImageChops.add(diff, diff, 2.0, -100)
        bbox = diff.getbbox()
        if bbox:
            return im.crop(bbox)

    buffer = trim(buffer)
    image_array = np.asarray(buffer)
    return image_array
