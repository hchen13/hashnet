import base64
from io import BytesIO

from PIL import Image

from db import DBManager
from engine import Engine
from settings import db_params
from utils import load_base64, display_image, load_image


def b64encode(image_path):
    with open(image_path, 'rb') as fin:
        data = fin.read()
    return base64.b64encode(data)


if __name__ == '__main__':
    engine = Engine()

    # image_path = "/Users/ethan/fishsaying/projects/image-search/test-images/2.png"
    image_path = '/Users/ethan/Pictures/datasets/marvel/black widow/pic_001.jpg'
    image = load_image(image_path)
    result, similarity = engine.search(image, 500)
    print(similarity)
    print(result.serialize())