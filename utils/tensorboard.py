import io

import cv2
import tensorflow as tf
from matplotlib import pyplot as plt


def visualize_matplotlib_figure_tensorboard(figure, tag):
    w, h = figure.canvas.get_width_height()

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    graph_image_summary = tf.Summary.Image(encoded_image_string=buffer.getvalue(), height=h, width=w)
    graph_summary = tf.Summary.Value(tag=tag, image=graph_image_summary)

    summary = tf.Summary(value=[graph_summary])
    figure.clear()
    return summary


def image_summary(img, tag):
    """Generate image summary for the given image numpy array."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, encoded_img = cv2.imencode('.jpg', img, encode_param)
    if not result:
        return None

    w, h = img.shape[:2]

    img_summary = tf.Summary.Image(encoded_image_string=encoded_img.tobytes(), height=h, width=w)
    summary_value = tf.Summary.Value(tag=tag, image=img_summary)
    summary = tf.Summary(value=[summary_value])
    return summary
