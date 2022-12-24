# Copyright (c) 2022 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import argparse

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

from utils import create_dataset


def create_parser():
    p = argparse.ArgumentParser(
        description="Train variational autoencoder on chicago face dataset."
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
        default="data/results/2022-12-24-09-29-45",
        help="Results location"
    )
    return p


def main(args):
    """
    Evaluate pretrained generator.

    Load pretrained encoder and decoder models,
    compute encodings space based on train data
    and generate some new images.
    """

    # load models

    encoder = tf.keras.models.load_model(os.path.join(args.results_dir, 'encoder'))
    decoder = tf.keras.models.load_model(os.path.join(args.results_dir, 'decoder'))

    # load images and compute encodings

    dataset = create_dataset(file_pattern=(args.data_dir + "/*")).batch(16)

    codes, _ = encoder.predict(dataset)
    codes_min = np.min(codes, axis=0)
    codes_max = np.max(codes, axis=0)

    # generate new images

    fig, ax = plt.subplots(
        nrows=5,
        ncols=10,
        dpi=150,
    )
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
    # plt.show()


if __name__ == "__main__":
    main(create_parser().parse_args())