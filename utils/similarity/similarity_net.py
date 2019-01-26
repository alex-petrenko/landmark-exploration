"""
This metric uses an encoder network as well as a smilarity network to determine how 'similar' two observations are.

"""
import tensorflow as tf

from algorithms.encoders import EncoderCNN
from algorithms.tf_utils import dense


class SimilarityNetwork:
    def __init__(self, ph_ob1, ph_ob2, name):
        self.name = name
        self._ob1 = ph_ob1
        self._ob2 = ph_ob2

        def make_encoder_cnn(ph_observations, image_enc_name, name):
            cnn = EncoderCNN(ph_observations, None, image_enc_name, name).encoded_input
            cnn = dense(cnn, 256)
            return cnn

        encoder_template = tf.make_template("encoder_template", make_encoder_cnn)

        left = encoder_template(self._ob1, 'convnet_simple', name)
        right = encoder_template(self._ob2, 'convnet_simple', name)
        out = tf.concat([left, right], 1)
        out = dense(out, 128)
        out = dense(out, 16)
        out = dense(out, 2, activation=None)
        self.logits = out

        self.labels = tf.argmax(self.logits)

        # print([v.name for v in tf.all_variables()])

    def pred(self, session, ob1, ob2):
        pred = session.run(
            [self.labels, self.logits],
            feed_dict={self._ob1: ob1, self._ob2: ob2},
        )
        return pred

    def similar(self, ob1, ob2):
        pass

    def train_step(self, ob1, ob2, steps_apart):
        pass
