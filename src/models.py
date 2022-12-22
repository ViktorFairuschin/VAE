import tensorflow as tf


class Sampling(tf.keras.layers.Layer):
    """
    Sample z from latent distribution.
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
    Closed form solution of Kullback-Leibler
    divergence between true and approximate posterior.
    """

    def call(self, mean, logvar):
        loss = - 0.5 * (1 + logvar - tf.square(mean) - tf.exp(logvar))
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        return loss

    def get_config(self):
        config = super(KullbackLeiblerDivergence, self).get_config()
        return config


class BinaryCrossentropy(tf.keras.losses.Loss):
    """
    Binary cross entropy loss used to calculate
    the expected likelihood.
    """

    def call(self, inputs, outputs):
        loss = tf.keras.losses.binary_crossentropy(inputs, outputs)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))
        return loss

    def get_config(self):
        config = super(BinaryCrossentropy, self).get_config()
        return config


class VariationalAutoencoder(tf.keras.models.Model):

    def __init__(
            self,
            img_shape,
            conv_filters,
            dense_units,
            activation,
            z_dim,
            beta,
            **kwargs
    ):
        super(VariationalAutoencoder, self).__init__(**kwargs)

        self.img_shape = img_shape
        self.conv_filters = conv_filters
        self.dense_units = dense_units
        self.activation = activation
        self.z_dim = z_dim
        self.beta = beta

        # init encoder network

        enc_input = tf.keras.layers.Input(
            shape=self.img_shape,
            name='encoder_input'
        )
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
        x = tf.keras.layers.Flatten(name='flatten')(x)
        for i, units in enumerate(self.dense_units):
            x = tf.keras.layers.Dense(
                units=units,
                activation=self.activation,
                name=f'encoder_dense_{i}',
            )(x)
        mean = tf.keras.layers.Dense(self.z_dim, name='mean')(x)
        logvar = tf.keras.layers.Dense(self.z_dim, name='logvar')(x)

        self.encoder = tf.keras.models.Model(
            inputs=enc_input,
            outputs=[mean, logvar],
            name='encoder'
        )

        # init decoder network

        _div = 2 ** len(self.conv_filters)
        _shape = (int(self.img_shape[0] / _div), int(self.img_shape[1] / _div), self.conv_filters[-1])

        dec_input = tf.keras.layers.Input(
            shape=self.z_dim,
            name='decoder_input'
        )
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
        dec_output = tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=3,
            strides=1,
            activation='sigmoid',
            padding='same',
            name=f'decoder_conv_{i + 1}',
        )(x)

        self.decoder = tf.keras.models.Model(
            inputs=dec_input,
            outputs=dec_output,
            name='decoder'
        )

        # init sampling layer

        self.sampling = Sampling()

        # set up losses and metrics

        self.kl_loss = KullbackLeiblerDivergence()
        self.bce_loss = BinaryCrossentropy()

        self.loss_metrics = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        self.kl_metrics = tf.keras.metrics.Mean('kl', dtype=tf.float32)
        self.bce_metrics = tf.keras.metrics.Mean('bce', dtype=tf.float32)

    def call(self, inputs, **kwargs):

        if isinstance(inputs, tuple):
            inputs, _ = inputs

        mean, logvar = self.encoder(inputs)
        sample = self.sampling([mean, logvar])
        outputs = self.decoder(sample)

        return outputs

    @property
    def metrics(self):
        return [
            self.loss_metrics,
            self.kl_metrics,
            self.bce_metrics,
        ]

    def train_step(self, inputs):
        with tf.GradientTape() as g:
            loss = self.compute_loss(inputs)
        gradients = g.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {
            'loss': self.loss_metrics.result(),
            'kl': self.kl_metrics.result(),
            'bce': self.bce_metrics.result()
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

        self.loss_metrics.update_state(loss)
        self.kl_metrics.update_state(kl_loss)
        self.bce_metrics.update_state(bce_loss)

        return loss

    @tf.function
    def encode(self, inputs):
        mean, logvar = self.encoder(inputs)
        return mean, logvar

    @tf.function
    def decode(self, inputs):
        outputs = self.decoder(inputs)
        return outputs
