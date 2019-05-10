from engine import hamming_dist, cosine_dist
from prototype import Hashnet
from utils import load_image

if __name__ == "__main__":
    p1 = load_image('/Users/ethan/Desktop/p1.png')
    p2 = load_image('/Users/ethan/Desktop/p2.png')

    print("loading model")
    engine = Hashnet()
    engine.load('models/hashnet.h5')

    print("extracting features")
    a = engine.extract_features(p1, p2)
    codes, features = a
    print(codes.shape, features.shape)

    hamming = hamming_dist(codes[[0]], codes[[1]])
    similarity = cosine_dist(features[[0]], features[[1]])
    print(hamming, similarity)