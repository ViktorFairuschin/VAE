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


class VariationalAutoencoder(tf.keras.models.Model):
    """
    Variational autoencoder model.

    Consists of an encoder network used to parametrize the
    approximate posterior distribution q(z|x), a sampling layer
    used to generate samples z from the approximate posterior
    distribution q(z|x), and a decoder network used to parametrize
    the likelihood p(x|z).

    Params:
        img_shape: Image shape.
        conv_filters: Number of filters in the convolution layers.
        dense_units: Number of units in the dense layers.
        activation: Activation function.
        z_dim: Latent space dimension.
        beta: Additional constraint on the KL term.
    """

    def __init__(
            self,
            img_shape=(184, 128, 1),
            conv_filters=None,
            dense_units=None,
            activation='relu',
            z_dim=10,
            beta=1.0,
            **kwargs
    ):
        super(VariationalAutoencoder, self).__init__(**kwargs)

        if dense_units is None:
            dense_units = [2048, 512, 128]
        if conv_filters is None:
            conv_filters = [16, 32, 64]

        self.img_shape = img_shape
        self.conv_filters = conv_filters
        self.dense_units = dense_units
        self.activation = activation
        self.z_dim = z_dim
        self.beta = beta / z_dim

        # encoder input

        enc_input = tf.keras.layers.Input(
            shape=self.img_shape,
            name='encoder_input'
        )

        # encoder convolution block

        x = enc_input
        for i, filters in enumerate(self.conv_filters):
            x = tf.keras.layers.Conv2D(
                filters=filters,
                kernel_size=3,
                strides=(2, 2),
                activation=self.activation,
                padding='same',
                name=f'encoder_conv_{i}',
            )(x)

        # encoder dense block

        x = tf.keras.layers.Flatten(name='flatten')(x)
        for i, units in enumerate(self.dense_units):
            x = tf.keras.layers.Dense(
                units=units,
                activation=self.activation,
                name=f'encoder_dense_{i}',
            )(x)

        # encoder outputs

        z_mean = tf.keras.layers.Dense(self.z_dim, name='z_mean')(x)
        z_logvar = tf.keras.layers.Dense(self.z_dim, name='z_logvar')(x)

        # encoder model

        self.encoder = tf.keras.models.Model(
            inputs=enc_input,
            outputs=[z_mean, z_logvar],
            name='encoder'
        )

        # sampling layer

        self.sampling = Sampling()

        # decoder input

        dec_input = tf.keras.layers.Input(
            shape=self.z_dim,
            name='decoder_input'
        )

        # decoder dense block

        _div = 2 ** len(self.conv_filters)
        _shape = (int(self.img_shape[0] / _div), int(self.img_shape[1] / _div), self.conv_filters[-1])

        x = dec_input
        for i, units in enumerate(self.dense_units.copy()[::-1]):
            x = tf.keras.layers.Dense(
                units=units,
                activation=self.activation,
                name=f'decoder_dense_{i}',
            )(x)

        x = tf.keras.layers.Dense(
            units=(_shape[0] * _shape[1] * _shape[2]),
            activation=self.activation,
            name=f'decoder_dense_{i + 1}'
        )(x)

        # decoder convolution block

        x = tf.keras.layers.Reshape(target_shape=_shape, name='reshape')(x)
        for i, filters in enumerate(self.conv_filters.copy()[::-1]):
            x = tf.keras.layers.Conv2DTranspose(
                filters=filters,
                kernel_size=3,
                strides=(2, 2),
                activation=self.activation,
                padding='same',
                name=f'decoder_conv_{i}',
            )(x)

        # decoder output

        dec_output = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=3,
            strides=1,
            activation='sigmoid',
            padding='same',
            name=f'decoder_conv_{i + 1}',
        )(x)

        # decoder model

        self.decoder = tf.keras.models.Model(
            inputs=dec_input,
            outputs=dec_output,
            name='decoder'
        )

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

    def get_config(self):
        config = super(VariationalAutoencoder, self).get_config()
        config.update({"img_shape": self.img_shape})
        config.update({"conv_filters": self.conv_filters})
        config.update({"dense_units": self.dense_units})
        config.update({"activation": self.activation})
        config.update({"z_dim": self.z_dim})
        config.update({"beta": self.beta})
        return config


# ------------------------------------------------------------- #
# ------------------------------------------------------------- #
# ------------------------------------------------------------- #


class ConvBlock(tf.keras.layers.Layer):
    """
    Convolution block consisting of a Conv2D and a MaxPooling2D layer.
    """

    def __init__(self, filters: int, activation: str, **kwargs) -> None:
        """
        :param filters: Number of filters in the convolution layer
        :param activation: Activation function
        :param kwargs: Additional keyword arguments
        """

        super().__init__(**kwargs)

        self.filters = filters
        self.activation = activation

        self.conv = tf.keras.layers.Conv2D(
            filters=self.filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation=self.activation,
            padding='same',
        )

        self.pool = tf.keras.layers.MaxPooling2D(
            pool_size=(4, 4)
        )

    def call(self, inputs, *args, **kwargs):
        x = self.conv(inputs)
        outputs = self.pool(x)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"activation": self.activation})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class ConvTransposeBlock(tf.keras.layers.Layer):
    """
    Transpose convolution block consisting of a Conv2DTranspose
    and an UpSampling2D layer.
    """

    def __init__(self, filters: int, activation: str, **kwargs) -> None:
        """
        :param filters: Number of filters in the convolution layer
        :param activation: Activation function
        :param kwargs: Additional keyword arguments
        """
        super().__init__(**kwargs)

        self.filters = filters
        self.activation = activation

        self.pool = tf.keras.layers.UpSampling2D(
            size=(4, 4)
        )

        self.conv = tf.keras.layers.Conv2DTranspose(
            filters=self.filters,
            kernel_size=(3, 3),
            strides=(2, 2),
            activation=self.activation,
            padding='same',
        )

    def call(self, inputs, *args, **kwargs):
        x = self.pool(inputs)
        outputs = self.conv(x)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"filters": self.filters})
        config.update({"activation": self.activation})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_encoder(activation: str, z_dim: int) -> tf.keras.Model:
    """
    Build encoder model.

    :param activation: Activation function
    :param z_dim: Latent space dimension

    :return: Encoder Model
    """

    enc_input = tf.keras.layers.Input(shape=(1024, 512, 1), name='encoder_input')

    x = enc_input
    for i, filters in enumerate([32, 64, 128]):
        x = ConvBlock(filters=filters, activation=activation, name=f'encoder_conv_{i}')(x)

    x = tf.keras.layers.Flatten(name='flatten')(x)
    for i, units in enumerate([128, 64, 32]):
        x = tf.keras.layers.Dense(units=units, activation=activation, name=f'encoder_dense_{i}')(x)

    z_mean = tf.keras.layers.Dense(z_dim, name='z_mean')(x)
    z_logvar = tf.keras.layers.Dense(z_dim, name='z_logvar')(x)

    encoder = tf.keras.models.Model(inputs=enc_input, outputs=[z_mean, z_logvar], name='encoder')

    return encoder


def build_decoder(activation: str, z_dim: int) -> tf.keras.Model:
    """
    Build decoder model.

    :param activation: Activation function
    :param z_dim: Latent space dimension

    :return: Decoder Model
    """

    dec_input = tf.keras.layers.Input(shape=z_dim, name='decoder_input')

    x = dec_input
    for i, units in enumerate([32, 64, 128, 256]):
        x = tf.keras.layers.Dense(units=units, activation=activation, name=f'decoder_dense_{i}')(x)

    x = tf.keras.layers.Reshape(target_shape=(2, 1, 128), name='reshape')(x)
    for i, filters in enumerate([128, 64, 32]):
        x = ConvTransposeBlock(filters=filters, activation=activation, name=f'decoder_conv_{i}')(x)

    dec_output = tf.keras.layers.Conv2DTranspose(
        filters=1, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid', padding='same', name=f'decoder_output'
    )(x)

    decoder = tf.keras.models.Model(inputs=dec_input, outputs=dec_output, name='decoder')

    return decoder


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

