import glob
import os

import numpy as np
from PIL import Image, ExifTags
from openpyxl import Workbook

from engine import hamming_dist, cosine_dist
from prototype import Hashnet
from utils import load_image, display_image
import openpyxl

# image_folder = "/Users/ethan/Pictures/datasets/成都博物馆/整理/"
image_folder = "/home/ethan/Pictures/整理/"
lib_dir = os.path.join(image_folder, '系统库')
query_dir = os.path.join(image_folder, '用户库')
lib_images = glob.glob(os.path.join(lib_dir, '*.JPG'))
query_images = glob.glob(os.path.join(query_dir, '*.JPG'))
hashnet = Hashnet()
hashnet.load()


def _rotate(_img):
    for orientation in ExifTags.TAGS.keys():
        if ExifTags.TAGS[orientation] == 'Orientation':
            break
    exif = dict(_img._getexif().items())

    if exif[orientation] == 3:
        _img = _img.rotate(180, expand=True)
    elif exif[orientation] == 6:
        _img = _img.rotate(270, expand=True)
    elif exif[orientation] == 8:
        _img = _img.rotate(90, expand=True)
    return _img


def encode(plist, batch_size=32):
    num_batches = len(plist) // batch_size
    code_list, feature_list = [], []
    for i in range(0, len(plist), batch_size):
        current_batch = i // batch_size + 1
        print("Feature extraction on batch #{}/{}...".format(current_batch, num_batches))
        paths = plist[i : i + batch_size]
        batch = [load_image(path) for path in paths]
        codes, features = hashnet.extract_features(*batch)
        code_list.append(codes)
        feature_list.append(features)
        print(codes.shape, features.shape)
        print("Done\n")
    codes = np.vstack(code_list)
    features = np.vstack(feature_list)
    print("Encoding complete! Got matrices {} and {}\n".format(codes.shape, features.shape))
    return codes, features


def main():
    if os.path.exists('lib.npz'):
        print("Loading library matrices from cache file...")
        with np.load('lib.npz') as data:
            lib_codes = data['arr_0']
            lib_features = data['arr_1']
        print("Loading complete\n")
    else:
        print("Extracting features from library images...")
        lib_codes, lib_features = encode(lib_images)
        np.savez('lib.npz', lib_codes, lib_features)

    if os.path.exists('query.npz'):
        print("Loading query matrices from cache file...")
        with np.load('query.npz') as data:
            query_codes = data['arr_0']
            query_features = data['arr_1']
        print("Loading complete\n")
    else:
        print("Extracting features from query images...")
        query_codes, query_features = encode(query_images)
        np.savez('query.npz', query_codes, query_features)


    def match(hash, features):
        hm = hamming_dist(hash, lib_codes)
        indices = np.argpartition(hm, 10)[:10]
        shortlist = lib_features[indices]
        dists = cosine_dist(features, shortlist)
        winner = indices[dists.argmin()]
        matched_image = lib_images[winner]
        return matched_image, dists.min()

    n = len(query_images)
    w = 200
    margin = 5
    width = (w + margin) * 2 + margin
    height = (w + margin) * n + margin
    canvas = Image.new('RGB', (width, height), (255, 255, 255))


    wb = Workbook()
    sheet = wb.active
    sheet.append(["待匹配图片", "命中图片", "算法相似度", "结果"])

    for i in range(n):
        print("Matching {}/{}...\n".format(i + 1, n))
        code, features = query_codes[i], query_features[i]
        matched_path, distance = match(code, features)

        query_image = _rotate(Image.open(query_images[i]))
        matched_image = _rotate(Image.open(matched_path))

        query_image.thumbnail((w, w))
        matched_image.thumbnail((w, w))

        x = margin
        y = i * (w + margin) + margin
        canvas.paste(query_image, (x, y))
        canvas.paste(matched_image, (x * 2 + w, y))

        p0 = os.path.join("用户库", os.path.basename(query_images[i]))
        p1 = os.path.join("系统库", os.path.basename(matched_path))

        sheet.append([p0, p1, 1 - distance])

    canvas.save("canvas.jpg", 'jpeg')
    wb.save("匹配结果.xlsx")


if __name__ == '__main__':
    main()
