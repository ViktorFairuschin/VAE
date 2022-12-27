# Copyright (c) 2022 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import pathlib

import tensorflow as tf


def parse_image(path):
    """Parse a single image."""
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[184, 128], preserve_aspect_ratio=False)
    return image


def create_dataset(file_pattern):
    """Create a dataset from images matching the provided pattern."""
    return tf.data.Dataset.list_files(file_pattern, shuffle=False).map(parse_image)


def create_train_dataset(file_pattern, batch_size=16):
    """Create train dataset from images matching the provided pattern."""
    def _preprocess(image):
        """Create input-target pairs."""
        return image, image

    ds = create_dataset(file_pattern)
    ds = ds.shuffle(2000).batch(batch_size).map(_preprocess).cache().prefetch(tf.data.AUTOTUNE)
    return ds


class KLWeightAnnealing(tf.keras.callbacks.Callback):
    """
    KL weight annealing callback.

    Params:
        epochs: Number of epochs required to increase KL weight from 0 to 1.
    """

    def __init__(self, epochs):
        super(KLWeightAnnealing, self).__init__()
        self.epochs = epochs

    def on_epoch_begin(self, epoch, logs=None):
        if self.epochs > 0:
            weight = min(epoch / self.epochs, 1)
            tf.keras.backend.set_value(self.model.kl_weight, weight)

