# Copyright (c) 2022 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf


class KullbackLeiblerDivergence(tf.keras.losses.Loss):
    """
    Closed form solution of Kullback-Leibler divergence
    between true posterior and approximate posterior.
    """

    def __init__(self, weight):
        """
        :param weight: KL weight used for warm up during training.
        """
        super(KullbackLeiblerDivergence, self).__init__()
        self.weight = weight

    def call(self, mean, logvar):
        loss = - 0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        return self.weight * loss

    def get_config(self):
        config = super(KullbackLeiblerDivergence, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class BinaryCrossentropy(tf.keras.losses.Loss):
    """
    Binary crossentropy loss used to calculate the
    expected likelihood.
    """

    def call(self, inputs, outputs):
        loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        return loss

    def get_config(self):
        config = super(BinaryCrossentropy, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)