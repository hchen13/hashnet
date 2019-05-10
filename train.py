import os

import tensorflow as tf
from keras import optimizers, metrics
from keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.generic_utils import to_list

from prototype import Hashnet
from utils import ensure_dir, Chart

if "MACHINE_ROLE" in os.environ and os.environ['MACHINE_ROLE'] == 'trainer':
    IMAGE_ROOT = "/home/ethan/Pictures/文化物_"
else:
    # IMAGE_ROOT = "/Users/ethan/Pictures/datasets/文化物_"
    IMAGE_ROOT = "/Users/ethan/datasets/marvel"

TRAIN_DATA_DIR = os.path.join(IMAGE_ROOT, 'train')
VALID_DATA_DIR = os.path.join(IMAGE_ROOT, 'valid')


def normalize_image(images):
    return images / 127.5 - 1


def restore_image(images):
    return (images + 1) / 2


def load():
    train_gen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0,
        height_shift_range=0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=normalize_image,
        validation_split=0.25
    )
    valid_gen = ImageDataGenerator(
        preprocessing_function=normalize_image,
    )

    train_batches = train_gen.flow_from_directory(
        TRAIN_DATA_DIR,
        target_size=(Hashnet.MIN_RESOLUTION, Hashnet.MIN_RESOLUTION),
        color_mode='rgb',
        batch_size=32,
        shuffle=True,
    )
    valid_batches = valid_gen.flow_from_directory(
        VALID_DATA_DIR,
        target_size=(Hashnet.MIN_RESOLUTION, Hashnet.MIN_RESOLUTION),
        color_mode='rgb',
        batch_size=32,
        shuffle=True
    )
    return train_batches, valid_batches, train_batches.num_classes


train_batches, valid_batches, num_classes = load()


def train(experiment_name, engine, learning_rate, epochs, eval_steps=None):
    model = engine.create_model(num_classes)
    optimizer = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc', metrics.top_k_categorical_accuracy])

    trial_name = '{}-{}'.format(engine.model_name, experiment_name)
    print('\n[info] initializing training session, code name: [{}]'.format(trial_name.upper()))

    logdir = os.path.join('./logs/', trial_name)
    train_chart = Chart(
        log_dir=os.path.join(logdir, 'train'),
        histogram_freq=0,
        write_graph=False,
        write_images=True
    )
    valid_chart = Chart(
        log_dir=os.path.join(logdir, 'validation'),
        histogram_freq=0,
        write_graph=False,
        write_images=True
    )

    def display_on_chart(outputs, chart, num):
        logs = {}
        outputs = to_list(outputs)
        for l, o in zip(model.metrics_names, outputs):
            logs[l] = o
        chart.draw(logs, num)

    train_chart.set_model(model)
    valid_chart.set_model(model)

    steps_per_epoch = len(train_batches)
    global_steps = 0
    min_acc = 70

    for ep in range(epochs):
        print("Epoch {}/{}".format(ep + 1, epochs))
        for batch_num in range(steps_per_epoch):
            x_batch, y_batch = train_batches[batch_num]
            global_steps += 1
            outs = model.train_on_batch(x_batch, y_batch)

            if batch_num % 10 == 0:
                progress = batch_num / steps_per_epoch * 100
                print("steps: {}/{} | progress: {:.2f}%".format(batch_num, steps_per_epoch, progress))

            if global_steps % eval_steps == 0:
                print('\n[info] validating...')
                val_outs = model.evaluate_generator(
                    valid_batches,
                    len(valid_batches),
                    workers=0,
                    verbose=1
                )
                display_on_chart(outs, train_chart, global_steps // eval_steps)
                display_on_chart(val_outs, valid_chart, global_steps // eval_steps)

        val_acc = val_outs[2] * 100
        if val_acc > min_acc:
            print('\n[info] saving intermediate model, rank-5 accuracy: {:.2f}%'.format(val_acc))
            ensure_dir('models')
            engine.save("models/tmp.h5")
            min_acc = val_acc

    ensure_dir('models')
    engine.save("models/final.h5")


if __name__ == '__main__':
    train_params = {
        'learning_rate': 0.0003,
        'epochs': 10,
        'eval_steps': 100
    }

    engine = Hashnet(
        hash_length=4096,
        pre_code_layers=0,
        post_code_layers=1,
        hidden_size=2048,
        keep_prob=.6
    )

    train(experiment_name='trial2', engine=engine, **train_params)
