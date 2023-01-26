# Copyright (c) 2022 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from models import VAE
from data import create_dataset


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser object for parsing command line strings.

    :return: Argument parser object
    """

    parser = argparse.ArgumentParser(description="Validate model.")

    parser.add_argument("--data_dir", type=str, default="data", help="Data location")
    parser.add_argument("--results_dir", type=str, default="results", help="Results location")

    return parser


def main(args: argparse.Namespace) -> None:
    """
    Validate model.

    :param args: Object holding arguments created by argument parser
    """

    # load model
    encoder = tf.keras.models.load_model(os.path.join(args.results_dir, 'encoder'))
    decoder = tf.keras.models.load_model(os.path.join(args.results_dir, 'decoder'))

    model = VAE(activation='relu', z_dim=10, beta=1.0)
    model.encoder = encoder
    model.decoder = decoder

    # load images
    dataset = create_dataset(file_pattern=(args.data_dir + "/*"))

    # generate reconstructions
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(20, 6))
    _ = [(ax.set_xticks([]), ax.set_yticks([])) for ax in axes.flatten()]

    for col, image in enumerate(dataset.take(10)):
        rec_image = model.predict(tf.expand_dims(image, axis=0))[0]

        axes[0][col].imshow(image.numpy(), cmap='gray')
        axes[1][col].imshow(rec_image, cmap='gray')

    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, 'reconstructed.png'))
    plt.show()

    # compute encodings
    dataset = dataset.batch(16)

    codes, _ = encoder.predict(dataset)
    codes_min = np.min(codes, axis=0)
    codes_max = np.max(codes, axis=0)

    # generate new images
    fig, ax = plt.subplots(nrows=5, ncols=10, dpi=150)
    for col in range(10):
        for row, value in enumerate(np.linspace(codes_min[col], codes_max[col], 5)):
            sample = np.zeros((1, 10))
            sample[0, row] = value
            decoded = decoder(sample)
            ax[row][col].imshow(decoded[0], cmap='gray')
            ax[row][col].set_xticks([])
            ax[row][col].set_yticks([])
            ax[row][col].set_title(f'z[{col}] = {value :.2f}', fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(args.results_dir, 'generated.png'))
    plt.show()


if __name__ == "__main__":
    main(create_parser().parse_args())
