import os

import numpy as np
from keras import Model
from keras.applications import VGG16, VGG19, InceptionV3, vgg16
from keras.layers import Dense, Dropout, regularizers, GlobalAveragePooling2D, BatchNormalization, Flatten
from keras.models import load_model
from keras.utils import plot_model


def binarize(vec):
    bin = (vec > .5) * 1
    return bin


BASE_DIR = os.path.dirname(__file__)


class Hashnet:

    BASE_MODEL = 'inceptionv3'
    MIN_RESOLUTION = 224
    IMAGE_SHAPE = (MIN_RESOLUTION, MIN_RESOLUTION, 3)
    HASH_LAYER = 'code'
    FEATURE_LAYER = 'features'


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
        pattern = '{base_model}_h{hash_length}xdense{hidden_size}'
        return pattern.format(
            base_model=self.BASE_MODEL,
            hash_length=self.hash_length,
            hidden_size=self.hidden_size,
        )

    def _get_base_model(self):
        params = {
            'weights': 'imagenet',
            'include_top': False,
            'input_shape': self.IMAGE_SHAPE,
            # 'pooling': 'avg',
        }
        if self.BASE_MODEL.lower() == 'vgg16':
            return VGG16(**params)
        elif self.BASE_MODEL.lower() == 'vgg19':
            return VGG19(**params)
        else:
            return InceptionV3(**params)

    def create_model(self, num_classes):
        """
        Creates a neural network architecture with the given hyper-params

        :param num_classes: number of output neurons of the neural network
        :param alpha: learning rate, defaults to 0.01
        :return: untrained architecture
        """
        l2_lambda = .01

        base_model = self._get_base_model()
        x = base_model.output

        bottleneck = GlobalAveragePooling2D(name=self.FEATURE_LAYER)(x)
        dense = Dropout(1 - self.keep_prob)(bottleneck)

        for i in range(self.pre_code_layers):
            dense = Dense(
                self.hidden_size,
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_lambda),
                name='pre{}'.format(i + 1)
            )(dense)
            dense = BatchNormalization()(dense)
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
                kernel_regularizer=regularizers.l2(l2_lambda),
                name='post{}'.format(i + 1)
            )(dense)
            dense = BatchNormalization()(dense)
            dense = Dropout(1 - self.keep_prob)(dense)

        predictions = Dense(
            num_classes,
            activation='softmax',
            kernel_regularizer=regularizers.l2(l2_lambda)
        )(dense)

        model = Model(inputs=base_model.inputs, outputs=predictions)
        for layer in base_model.layers:
            if layer.name == 'mixed5':
                break
            layer.trainable = False

        self.model = model
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
    # plot_model(hashnet.model, show_layer_names=True, show_shapes=True)
    hashnet.model.summary()
