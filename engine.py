from datetime import datetime

import numpy as np

from db import DBManager, Image
from prototype import Hashnet
from settings import db_params


def hamming_dist(vec1, vec2):
    iden = (vec1 != vec2) * 1
    return iden.sum(axis=1)


def euclidean_dist(vec, mat):
    dists = np.square(vec - mat).sum(axis=1)
    return np.sqrt(dists)


def cosine_dist(vec, mat):
    numerator = np.dot(vec, mat.T)
    vec_norm = np.sqrt(np.square(vec).sum())
    mat_norm = np.sqrt(np.square(mat).sum(axis=1))
    denominator = vec_norm * mat_norm
    similarities = numerator / denominator
    return 1 - similarities


def get_hash_matrix(candidates):
    hash_list = []
    for i, can in enumerate(candidates):
        hash_list.append(can.get_hash())
    return np.array(hash_list)


def get_feature_matrix(candidates):
    vec_list = []
    for can in candidates:
        vec_list.append(can.get_features())
    return np.array(vec_list)


class Engine:
    def __init__(self):
        self.hashnet = Hashnet()
        self.hashnet.load()
        self.db = DBManager(**db_params)

    def match(self, target, candidates, threshold=10):
        hash_matrix = get_hash_matrix(candidates)
        hashcode, features = target

        hamming_dists = hamming_dist(hashcode, hash_matrix)
        indices = np.argpartition(hamming_dists, threshold)[:threshold]
        shortlist = [candidates[i] for i in indices]

        feature_matrix = get_feature_matrix(shortlist)
        dists = cosine_dist(features, feature_matrix)
        winner = shortlist[dists.argmin()]
        distance = dists.min()
        return winner, distance

    def search(self, target_image, page_size):
        tick = datetime.now()
        target_features = self.hashnet.extract_features(target_image)
        session = self.db.Session()
        current_page = 0
        queryset = session.query(Image).filter(Image.hash != None)
        optimal_dist = None
        winner = None
        while True:
            query = queryset.limit(page_size).offset(current_page * page_size)
            candidates = query.all()
            if not len(candidates):
                break
            current_page += 1
            matched, distance = self.match(target_features, candidates)
            if optimal_dist is None or distance < optimal_dist:
                optimal_dist = distance
                winner = matched
        session.close()
        tock = datetime.now()
        print("\nSearching completed in {}".format(tock - tick), flush=True)
        if winner is None:
            print("There is nothing found during the search, most likely because there's not images in the database", flush=True)
            return None, None
        print("Most similar image is {}\nwith cosine distance={}".format(winner, optimal_dist), flush=True)
        return winner, 1 - optimal_dist