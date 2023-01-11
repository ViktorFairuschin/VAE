# Copyright (c) 2022 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import tensorflow as tf
from keras.layers import (
    Input,
    Dense,
    Reshape,
    Conv2D,
    MaxPooling2D,
    GlobalMaxPooling2D,
    Conv2DTranspose,
    UpSampling2D
)

from layers import Sampling
from losses import KullbackLeiblerDivergence, BinaryCrossentropy


def build_encoder(activation: str, z_dim: int) -> tf.keras.Model:
    """
    Build encoder model.

    :param activation: Activation function
    :param z_dim: Latent space dimension

    :return: Encoder model
    """

    inputs = Input(shape=(256, 128, 1))
    x = Conv2D(filters=32, kernel_size=3, strides=2, activation=activation, padding='same')(inputs)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=64, kernel_size=3, strides=2, activation=activation, padding='same')(x)
    x = MaxPooling2D(pool_size=2)(x)
    x = Conv2D(filters=128, kernel_size=3, strides=2, activation=activation, padding='same')(x)
    x = GlobalMaxPooling2D()(x)
    x = Dense(units=128, activation=activation)(x)
    x = Dense(units=64, activation=activation)(x)
    x = Dense(units=32, activation=activation)(x)
    z_mean = Dense(z_dim)(x)
    z_logvar = Dense(z_dim)(x)

    return tf.keras.models.Model(inputs=inputs, outputs=[z_mean, z_logvar], name='encoder')


def build_decoder(activation: str, z_dim: int) -> tf.keras.Model:
    """
    Build decoder model.

    :param activation: Activation function
    :param z_dim: Latent space dimension

    :return: Decoder model
    """

    inputs = Input(shape=z_dim)
    x = Dense(units=32, activation=activation)(inputs)
    x = Dense(units=64, activation=activation)(x)
    x = Dense(units=128, activation=activation)(x)
    x = Dense(units=4096, activation=activation)(x)
    x = Reshape(target_shape=(8, 4, 128))(x)
    x = Conv2DTranspose(filters=128, kernel_size=3, strides=2, activation=activation, padding='same')(x)
    x = UpSampling2D(size=2)(x)
    x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, activation=activation, padding='same')(x)
    x = UpSampling2D(size=2)(x)
    x = Conv2DTranspose(filters=32, kernel_size=3, strides=2, activation=activation, padding='same')(x)
    outputs = Conv2DTranspose(filters=1, kernel_size=3, strides=1, activation='sigmoid', padding='same')(x)

    return tf.keras.models.Model(inputs=inputs, outputs=outputs, name='decoder')


class VAE(tf.keras.models.Model):
    """
    Variational autoencoder.

    Consists of an encoder network used to parametrize the
    approximate posterior distribution q(z|x), a sampling layer
    used to generate samples z from the approximate posterior
    distribution q(z|x), and a decoder network used to parametrize
    the likelihood p(x|z).
    """

    def __init__(self, activation: str, z_dim: int, beta: float, **kwargs) -> None:
        """
        :param activation: Activation function
        :param z_dim: Latent space dimension
        :param beta: A constant used to control the tradeoff between the KL and reconstruction loss.
        :param kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.activation = activation
        self.z_dim = z_dim
        self.beta = beta / z_dim

        self.encoder = build_encoder(self.activation, self.z_dim)
        self.decoder = build_decoder(self.activation, self.z_dim)
        self.sampling = Sampling()

        # losses and metrics

        self.kl_weight = tf.Variable(1.0, trainable=False, name='kl_weight')
        self.kl_loss = KullbackLeiblerDivergence(self.kl_weight)
        self.bce_loss = BinaryCrossentropy()

        self.loss_metric = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.kl_metric = tf.keras.metrics.Mean('kl', dtype=tf.float32)
        self.bce_metric = tf.keras.metrics.Mean('bce', dtype=tf.float32)

    def call(self, inputs, **kwargs):

        if isinstance(inputs, tuple):
            inputs, _ = inputs

        z_mean, z_logvar = self.encoder(inputs)
        sample = self.sampling([z_mean, z_logvar])
        outputs = self.decoder(sample)

        return outputs

    @property
    def metrics(self):
        return [
            self.loss_metric,
            self.kl_metric,
            self.bce_metric,
        ]

    def train_step(self, inputs):
        with tf.GradientTape() as g:
            loss = self.compute_loss(inputs)
        gradients = g.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {
            'loss': self.loss_metric.result(),
            'kl': self.kl_metric.result(),
            'bce': self.bce_metric.result(),
        }

    @tf.function
    def compute_loss(self, inputs):
        if isinstance(inputs, tuple):
            inputs, _ = inputs

        mean, logvar = self.encoder(inputs)
        sample = self.sampling([mean, logvar])
        outputs = self.decoder(sample)

        kl_loss = self.beta * self.kl_loss(mean, logvar)
        bce_loss = self.bce_loss(inputs, outputs)
        loss = kl_loss + bce_loss

        self.loss_metric.update_state(loss)
        self.kl_metric.update_state(kl_loss)
        self.bce_metric.update_state(bce_loss)

        return loss
