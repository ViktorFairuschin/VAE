# Copyright (c) 2022 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import datetime

import tensorflow as tf

from models import VAE
from data import create_dataset
from callbacks import WarmUp


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser object for parsing command line strings.

    :return: Argument parser object
    """

    parser = argparse.ArgumentParser(description="Train model.")

    parser.add_argument("--activation", type=str, default="relu", help="Activation function")
    parser.add_argument("--z_dim", type=int, default=10, help="Latent space dimension")
    parser.add_argument("--beta", type=float, default=1.0, help="Beta parameter in VAE loss")

    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--optimizer", type=str, default="adam", help="Optimizer")
    parser.add_argument("--lr", type=float, default=1.0e-3, help="Learning rate")
    parser.add_argument("--warmup", type=int, default=0, help="Number of warm up epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--data_dir", type=str, default="data/chicago-face-database", help="Data location")
    parser.add_argument("--results_dir", type=str, default="data/results", help="Results location")

    return parser


def main(args: argparse.Namespace) -> None:
    """
    Train model.

    :param args: Object holding arguments created by argument parser
    """

    # set random seed
    tf.random.set_seed(args.seed)

    # create train dataset
    train_ds = create_dataset(file_pattern=(args.data_dir + "/*")).batch(args.batch_size)

    # create model
    model = VAE(activation=args.activation, z_dim=args.z_dim, beta=args.beta)

    # set up optimizer
    # optimizer = tf.keras.optimizers.deserialize(args.optimizer)
    # tf.keras.optimizers.Adam(learning_rate=args.lr)

    # create results dir
    results_dir = os.path.join(args.results_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.mkdir(results_dir)

    # set up callbacks
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(results_dir, 'logs.csv'), separator=',', append=True)
    warmup = WarmUp(epochs=args.warmup)
    terminate_on_nan = tf.keras.callbacks.TerminateOnNaN()

    # compile and fit model
    model.compile(optimizer=args.optimizer)
    model.fit(
        x=train_ds, y=None,
        epochs=args.epochs,
        initial_epoch=0,
        callbacks=[
            csv_logger,
            terminate_on_nan,
            warmup,
        ]
    )

    # save trained models
    model.encoder.save(filepath=os.path.join(results_dir, 'encoder'))
    model.decoder.save(filepath=os.path.join(results_dir, 'decoder'))


if __name__ == "__main__":
    main(create_parser().parse_args())
