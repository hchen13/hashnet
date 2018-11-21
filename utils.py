import base64
import os
from io import BytesIO

import numpy as np
from PIL import Image

from prototype import Hashnet

def display_image(*images, title='image display', col=None):
    from matplotlib import pyplot as plt
    if col is None:
        col = len(images)
    plt.figure(figsize=(16, 9))
    plt.title(title)
    row = np.math.ceil(len(images) / col)
    for i, image in enumerate(images):
        plt.subplot(row, col, i + 1)
        plt.imshow(image, cmap='gray')
    plt.show()


def load_base64(encoded, dim=Hashnet.MIN_RESOLUTION):
    return load_raw(base64.b64decode(encoded), dim)


def load_raw(bytes, dim=Hashnet.MIN_RESOLUTION):
    image = Image.open(BytesIO(bytes))
    width, height = image.size
    size = min(width, height)
    if size < dim:
        return None
    image = image.crop((
        (width - size) // 2,
        (height - size) // 2,
        (width + size) // 2,
        (height + size) // 2
    ))
    if size > dim:
        image.thumbnail((dim, dim), Image.ANTIALIAS)

    image_array = np.array(image) / 255
    if image_array.shape[2] != 3:
        image_array = image_array[:, :, :3]
    return image_array


def load_image(path, dim=Hashnet.MIN_RESOLUTION):
    image = Image.open(path)
    width, height = image.size
    size = min(width, height)
    image = image.crop((
        (width - size) // 2,
        (height - size) // 2,
        (width + size) // 2,
        (height + size) // 2
    ))
    if size > dim:
        image.thumbnail((dim, dim), Image.ANTIALIAS)
    array = np.array(image) / 255
    return array[:, :, :3]


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
