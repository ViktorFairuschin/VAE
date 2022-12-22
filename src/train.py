import argparse

import tensorflow as tf

from models import VariationalAutoencoder
from utils import create_train_dataset


def create_parser():
    p = argparse.ArgumentParser(
        description="Train variational autoencoder on chicago face dataset."
    )
    p.add_argument(
        "--conv_filters",
        type=list,
        default=[16, 32, 64],
        help="Number of output filters in the convolution layers"
    )
    p.add_argument(
        "--dense_units",
        type=list,
        default=[2048, 512, 128],
        help="Number of units in dense layers"
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
    p.add_argument(
        "--img_path",
        type=str,
        default="data/chicago-face-database/*",
        help="Train images location"
    )
    p.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Log dir"
    )
    return p


def main(args):

    # create train dataset

    train_ds = create_train_dataset(
        path=args.img_path,
        batch_size=args.batch_size
    )

    # create model

    model = VariationalAutoencoder(
        img_shape=(184, 128, 1),
        conv_filters=args.conv_filters,
        dense_units=args.dense_units,
        activation=args.activation,
        z_dim=args.z_dim,
        beta=args.beta,
    )

    # set up optimizer

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # set up callbacks

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=args.log_dir)

    # compile and fit model

    model.compile(optimizer=optimizer)
    model.fit(
        train_ds,
        epochs=args.epochs,
        callbacks=[tensorboard]
    )


if __name__ == "__main__":
    main(create_parser().parse_args())
