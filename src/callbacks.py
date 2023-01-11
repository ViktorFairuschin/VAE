# Copyright (c) 2022 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf


class WarmUp(tf.keras.callbacks.Callback):

    def __init__(self, epochs: int):
        """
        Warm up the model by slowly increasing the relative weight
        of the KL term at the beginning of the training.

        :param epochs: Number of warmup epochs
        """

        super().__init__()
        self.epochs = epochs

    def on_epoch_begin(self, epoch, logs=None):
        if self.epochs > 0:
            weight = min(epoch / self.epochs, 1)
            tf.keras.backend.set_value(self.model.kl_weight, weight)
