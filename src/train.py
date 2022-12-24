# Copyright (c) 2022 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import argparse
import datetime

import tensorflow as tf

from model import VariationalAutoencoder
from utils import create_train_dataset


def create_parser():
    p = argparse.ArgumentParser(
        description="Train variational autoencoder on chicago face dataset."
    )

    # model parameters

    p.add_argument(
        "--conv_filters",
        type=list,
        default=[16, 32, 64],
        help="Number of filters in the convolution layers"
    )
    p.add_argument(
        "--dense_units",
        type=list,
        default=[2048, 512, 128],
        help="Number of units in the dense layers"
    )
    p.add_argument(
        "--activation",
        type=str,
        default="relu",
        help="Activation function to use"
    )
    p.add_argument(
        "--z_dim",
        type=int,
        default=10,
        help="Latent space dimension"
    )
    p.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Additional constraint on the KL term"
    )

    # training parameters

    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train the model"
    )
    p.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size"
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1.0e-3,
        help="Learning rate of the optimizer"
    )

    # inputs and outputs locations

    p.add_argument(
        "--inputs_dir",
        type=str,
        default="data/chicago-face-database",
        help="Inputs location"
    )
    p.add_argument(
        "--outputs_dir",
        type=str,
        default="data/results",
        help="Outputs location"
    )
    return p


def main(args):
    """
    Train variational autoencoder.
    """

    img_shape = (184, 128, 1)

    # create train dataset

    train_ds = create_train_dataset(
        file_pattern=(args.inputs_dir + "/*"),
        batch_size=args.batch_size
    )

    # create model

    model = VariationalAutoencoder(
        img_shape=img_shape,
        conv_filters=args.conv_filters,
        dense_units=args.dense_units,
        activation=args.activation,
        z_dim=args.z_dim,
        beta=args.beta,
    )

    # set up optimizer

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # set up csv logger

    outputs_dir = args.outputs_dir
    name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    os.mkdir(os.path.join(outputs_dir, name))

    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(outputs_dir, name, 'logs.csv'),
        separator=',',
        append=True
    )

    # compile and fit model

    terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()

    model.compile(optimizer=optimizer)
    model.fit(
        train_ds,
        epochs=args.epochs,
        callbacks=[
            csv_logger,
            terminate_on_nan,
        ]
    )

    model.encoder.save(os.path.join(outputs_dir, name, 'encoder'))
    model.decoder.save(os.path.join(outputs_dir, name, 'decoder'))


if __name__ == "__main__":
    main(create_parser().parse_args())