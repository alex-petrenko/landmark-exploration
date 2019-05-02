import random
from unittest import TestCase

import numpy as np
import tensorflow as tf

from algorithms.architectures.resnet_keras import ResnetBuilder
from algorithms.tests.test_wrappers import TEST_ENV_NAME
from utils.envs.doom.doom_utils import make_doom_env, doom_env_by_name
from utils.utils import log


class TestResnet(TestCase):
    # noinspection PyMethodMayBeStatic
    def skipped_test_resnet(self):
        # shape = (3 * 3, 84, 84)
        shape = (3, 84, 84)

        with tf.variable_scope('reach'):
            resnet = ResnetBuilder.build_resnet_18(shape, 2)
            adam = tf.train.AdamOptimizer(learning_rate=1e-4, name='loco_opt')
            resnet.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        num_data = 1000
        half_data = num_data // 2
        iterations = 3
        num_epochs_per_iteration = 5
        epoch = 0

        for i in range(iterations):
            log.info('Iteration %d...', i)

            x = np.empty([num_data, 84, 84, 3], dtype=np.float32)
            y = np.empty([num_data, 2], dtype=np.int32)

            for j in range(half_data):
                x[j, :, :, 0] = 1.0
                x[j, :, :, 1] = random.random()
                x[j, :, :, 2] = 0.0
                y[j] = [1, 0]

                x[half_data + j, :, :, 0] = 0.0
                x[half_data + j, :, :, 1] = random.random()
                x[half_data + j, :, :, 2] = 1.0
                y[half_data + j] = [0, 1]

            train_until = epoch + num_epochs_per_iteration
            resnet.fit(x, y, batch_size=64, epochs=train_until, verbose=1, initial_epoch=epoch)
            epoch += num_epochs_per_iteration

        x = np.empty([num_data, 84, 84, 3], dtype=np.float32)
        x[:half_data, :, :, 0] = 1.0
        x[:half_data, :, :, 1] = random.random()
        x[:half_data, :, :, 2] = 0.0
        x[half_data:, :, :, 0] = 0.0
        x[half_data:, :, :, 1] = random.random()
        x[half_data:, :, :, 2] = 1.0

        result = resnet.predict(x, verbose=1, batch_size=1024)
        log.info('result %r', result)

        env = make_doom_env(doom_env_by_name(TEST_ENV_NAME))
        obs = env.reset()
        env.close()
