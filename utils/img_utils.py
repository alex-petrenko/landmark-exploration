import numpy as np


def crop_map_image(image_array):
    """Uses Python Image Library (PIL) to crop the borders of an image"""
    # TODO: Use openCV instead of PIL?
    if image_array is None:
        return None
    from PIL import Image, ImageChops
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
