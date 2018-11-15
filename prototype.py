import os

import numpy as np
from keras import Model, optimizers
from keras.applications import VGG16, VGG19, InceptionV3
from keras.layers import Dense, Dropout, Flatten
from keras.models import load_model


def binarize(vec):
    bin = (vec > .5) * 1
    return bin


BASE_DIR = os.path.dirname(__file__)


class Hashnet:

    BASE_MODEL = 'vgg16'
    MIN_RESOLUTION = 160
    IMAGE_SHAPE = (MIN_RESOLUTION, MIN_RESOLUTION, 3)
    HASH_LAYER = 'code'
    FEATURE_LAYER = 'feature'


    def __init__(self, hash_length=1024, pre_code_layers=1, post_code_layers=0, hidden_size=256, keep_prob=.7):
        """
        Instantiate a Hashnet controller

        :param hash_length: total number of bits in the binary hash code
        :param pre_code_layers: number of dense layers before the latent hash layer
        :param hidden_size: number of neurons in each dense layer
        :param keep_prob: keep probability for dropout layers
        """
        self.hash_length = hash_length
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.pre_code_layers = pre_code_layers
        self.post_code_layers = post_code_layers
        self.model = None

    @property
    def model_name(self):
        return '{base_model}+hash{hash_length}/{hidden_size}x({pre}{post})_{keep_prob}'.format(
            base_model=self.BASE_MODEL,
            hash_length=self.hash_length,
            hidden_size=self.hidden_size,
            pre=self.pre_code_layers,
            post="+{}".format(self.post_code_layers) if self.post_code_layers else "",
            keep_prob=self.keep_prob
        )

    def _get_base_model(self):
        params = {
            'weights': 'imagenet',
            'include_top': False,
            'input_shape': self.IMAGE_SHAPE
        }
        if self.BASE_MODEL.lower() == 'vgg16':
            return VGG16(**params)
        elif self.BASE_MODEL.lower() == 'vgg19':
            return VGG19(**params)
        else:
            return InceptionV3(**params)

    def create_model(self, num_classes, alpha=1e-2):
        """
        Creates a neural network architecture with the given hyper-params

        :param num_classes: number of output neurons of the neural network
        :param alpha: learning rate, defaults to 0.01
        :return: untrained architecture
        """
        base_model = self._get_base_model()
        x = base_model.output
        # bottleneck = GlobalAveragePooling2D()(x)
        # bottleneck = Lambda(lambda y: squeeze(y, [1, 2]))(x)
        bottleneck = Flatten(name='feature')(x)

        dense = Dropout(1 - self.keep_prob)(bottleneck)

        for i in range(self.pre_code_layers):
            dense = Dense(
                self.hidden_size,
                activation='relu',
                name='pre{}'.format(i + 1)
            )(dense)
            dense = Dropout(1 - self.keep_prob)(dense)

        code = Dense(
            self.hash_length,
            activation='sigmoid',
            name='code'
        )(dense)
        code = Dropout(1 - self.keep_prob)(code)

        dense = code

        for i in range(self.post_code_layers):
            dense = Dense(
                self.hidden_size,
                activation='relu',
                name='post{}'.format(i + 1)
            )(dense)
            dense = Dropout(1 - self.keep_prob)(dense)

        predictions = Dense(num_classes, activation='softmax')(dense)

        model = Model(inputs=base_model.inputs, outputs=predictions)
        for layer in base_model.layers[:]:
            layer.trainable = False

        optimizer = optimizers.Adam(lr=alpha)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])
        self.model = model
        self.learning_rate = alpha
        return model

    def save(self, path):
        self.model.save(path)

    def load(self, path=os.path.join(BASE_DIR, 'models/hashnet.h5')):
        base = load_model(path)
        layers = [base.get_layer(name).output for name in [self.HASH_LAYER, self.FEATURE_LAYER]]
        self.model = Model(inputs=base.inputs, outputs=layers)
        self.model._make_predict_function()

    def extract_features(self, *images):
        """
        extracts images' binary hash codes and feature vectors

        :param images: arbitrary number of images represented as ndarrays
        :return: 2 ndarrays representing hash codes and features
        """
        outputs = self.model.predict(np.array(images))
        return binarize(outputs[0]), outputs[1]


if __name__ == '__main__':
    params = {
        'hash_length': 4096,
        'hidden_size': 1024,
        'pre_code_layers': 1,
        'post_code_layers': 1,
        'keep_prob': .7
    }
    hashnet = Hashnet(**params)
    hashnet.create_model(30)
    hashnet.model.summary()
