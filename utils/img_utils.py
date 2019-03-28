import cv2

import tensorflow as tf


def crop_map_image(image_array):
    """Uses Python Image Library (PIL) to crop the borders of an image"""
    # TODO: Use openCV instead of PIL?
    if image_array is None:
        return None
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


def image_summary(img, tag):
    """Generate image summary for the given image numpy array."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, encoded_img = cv2.imencode('.jpg', img, encode_param)
    if not result:
        return None

    w, h = img.shape[:2]

    img_summary = tf.Summary.Image(encoded_image_string=encoded_img.tobytes(), height=h, width=w)
    summary = tf.Summary.Value(tag=tag, image=img_summary)
    return summary
