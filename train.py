import os

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator

from prototype import Hashnet
from utils import ensure_dir

if "MACHINE_ROLE" in os.environ and os.environ['MACHINE_ROLE'] == 'trainer':
    IMAGE_ROOT = "/home/ethan/Pictures/文化物_"
else:
    IMAGE_ROOT = "/Users/ethan/Pictures/datasets/文化物_"


TRAIN_DATA_DIR = os.path.join(IMAGE_ROOT, 'train')
VALID_DATA_DIR = os.path.join(IMAGE_ROOT, 'valid')


epochs = 200


def load():
    idg = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_epsilon=1e-6,
        zca_whitening=False,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=lambda x: x / 255,
        data_format=None,
        validation_split=0.25
    )

    train_batches = idg.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(Hashnet.MIN_RESOLUTION, Hashnet.MIN_RESOLUTION),
        color_mode='rgb',
        batch_size=32,
        shuffle=True,
    )
    valid_batches = idg.flow_from_directory(
        VALID_DATA_DIR,
        target_size=(Hashnet.MIN_RESOLUTION, Hashnet.MIN_RESOLUTION),
        color_mode='rgb',
        batch_size=32,
        shuffle=True
    )
    return train_batches, valid_batches, train_batches.num_classes


train_batches, valid_batches, num_classes = load()


class History(TensorBoard):
    
    def __init__(self, *args, **kwargs):
        super(History, self).__init__(*args, **kwargs)
        self.global_steps = 0

    def write_log(self, names, logs, num):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.writer.add_summary(summary, num)
            self.writer.flush()

    def on_batch_end(self, _, logs=None):
        train_loss = logs['loss']
        train_acc = logs['acc']
        self.write_log(['loss/train', 'acc/train'], [train_loss, train_acc], self.global_steps)
        self.global_steps += 1

    def on_epoch_end(self, epoch, logs=None):
        self.write_log(['loss/validation', 'acc/validation'], [logs['val_loss'], logs['val_acc']], epoch)


def train(hash_length, pre_layers, post_layers, hidden_size, keep_prob, learning_rate):
    hashnet = Hashnet(
        hash_length,
        pre_code_layers=pre_layers,
        post_code_layers=post_layers,
        hidden_size=hidden_size,
        keep_prob=keep_prob
    )
    hashnet.create_model(num_classes, learning_rate)
    logdir = os.path.join('./logs', hashnet.model_name)
    tb = History(log_dir=logdir, histogram_freq=0, write_graph=False, write_images=True)

    hashnet.model.fit_generator(
        train_batches,
        epochs=epochs,
        validation_data=valid_batches,
        callbacks=[tb]
    )
    ensure_dir('models')
    hashnet.save("models/final.h5")


if __name__ == '__main__':

    # for pre in [0, 1, 2]:
    #     for post in [0, 1, 2]:
    #         if pre + post == 0:
    #             continue
    #         train(4096, pre, post, 2048, .7, learning_rate=1e-3)

    train_params = {
        'hash_length': 4096,
        'hidden_size': 2048,
        'pre_layers': 2,
        'post_layers': 1,
        'keep_prob': 1,
        'learning_rate': 1e-3
    }
    train(**train_params)
