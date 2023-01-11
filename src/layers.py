# Copyright (c) 2022 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    """
    Sample z from latent distribution q(z|x).
    """

    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        mean, logvar = inputs
        eps = tf.random.normal(shape=tf.shape(mean))
        z = eps * tf.exp(logvar * .5) + mean
        return z

    def get_config(self):
        config = super(Sampling, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

