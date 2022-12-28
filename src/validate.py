# Copyright (c) 2022 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import argparse

import tensorflow as tf

import matplotlib.pyplot as plt

from utils import create_dataset
from model import VariationalAutoencoder


def create_parser():
    p = argparse.ArgumentParser(
        description="Evaluate pretrained model."
    )

    # data and results locations

    p.add_argument(
        "--data_dir",
        type=str,
        default="data/chicago-face-database",
        help="Data location"
    )
    p.add_argument(
        "--results_dir",
        type=str,
        default="data/results/2022-12-28-13-52-54",
        help="Results location"
    )
    return p


def main(args):
    """
    Evaluate pretrained model.

    Load pretrained model and plot reconstructions.
    """

    # load model

    model = VariationalAutoencoder()
    model.encoder = tf.keras.models.load_model(os.path.join(args.results_dir, 'encoder'))
    model.decoder = tf.keras.models.load_model(os.path.join(args.results_dir, 'decoder'))

    # load images

    dataset = create_dataset(file_pattern=(args.data_dir + "/*")).shuffle(1000)

    # generate new images

    fig, ax = plt.subplots(
        nrows=2,
        ncols=10,
        dpi=150,
        figsize=(20, 6)
    )
    for col in range(10):
        img = next(iter(dataset))
        res = model.predict(tf.expand_dims(img, axis=0))[0]

        ax[0][col].imshow(img.numpy(), cmap='gray')
        ax[0][col].set_xticks([])
        ax[0][col].set_yticks([])
        ax[0][col].set_title('original', fontsize=6)

        ax[1][col].imshow(res, cmap='gray')
        ax[1][col].set_xticks([])
        ax[1][col].set_yticks([])
        ax[1][col].set_title('reconstruction', fontsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, 'reconstructed.png'))
    plt.show()


if __name__ == "__main__":
    main(create_parser().parse_args())
