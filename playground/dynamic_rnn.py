import time

import numpy as np
import sys

import tensorflow as tf


RNN_SIZE = 64
MAX_HORIZON = 10
DATA_SIZE = 3


def generate_random_data(batch_size):
    data = np.ones([batch_size, MAX_HORIZON, DATA_SIZE], dtype=np.float32)
    seq_len = [np.random.randint(0, MAX_HORIZON) for _ in range(batch_size)]
    answers = [l * 2 for l in seq_len]
    return data, seq_len, answers


def make_decoder(rnn_output):
    x = tf.layers.dense(rnn_output, 32, activation=tf.nn.relu)
    return tf.layers.dense(x, 1, activation=None)


def main():
    ph_data = tf.placeholder(tf.float32, shape=[None, MAX_HORIZON, DATA_SIZE])
    ph_len = tf.placeholder(tf.int32, shape=[None])
    ph_answers = tf.placeholder(tf.float32, shape=[None])

    cell = tf.nn.rnn_cell.GRUCell(RNN_SIZE)

    outputs, last_states = tf.nn.dynamic_rnn(cell=cell, inputs=ph_data, sequence_length=ph_len, dtype=tf.float32)
    decoder = tf.make_template('decoder', make_decoder, create_scope_now_=True)

    outputs_decoded = tf.squeeze(decoder(last_states), axis=1)

    loss = tf.losses.mean_squared_error(outputs_decoded, ph_answers)
    opt = tf.train.AdamOptimizer(learning_rate=1e-4)
    train = opt.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        num_batch = 20000
        for batch_i in range(num_batch):
            data, seq_len, answers = generate_random_data(batch_size=32)
            l, _ = sess.run([loss, train], feed_dict={ph_data: data, ph_len: seq_len, ph_answers: answers})
            if batch_i % 100 == 0:
                print('Batch:', batch_i, 'Loss:', l)

        # evaluate
        while True:
            data, seq_len, answers = generate_random_data(batch_size=1)
            prediction = sess.run(outputs_decoded, feed_dict={ph_data: data, ph_len: seq_len})
            print('Prediction:', prediction[0], 'correct answer:', answers[0])
            time.sleep(0.5)

    return 0


if __name__ == '__main__':
    sys.exit(main())
