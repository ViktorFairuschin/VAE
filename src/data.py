# Copyright (c) 2022 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
import tensorflow as tf


def process_image(path: str, output_shape: tuple = (256, 128, 1)) -> tf.Tensor:
    """
    Process a single image by applying the following steps:
    read file -> decode -> convert dtype > scale > resize

    :param path: Path to the image
    :param output_shape: Shape of the output tensor
    :return: A tensor with desired shape
    """

    height, width, channels = output_shape

    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[height, width], preserve_aspect_ratio=False)

    return image


def create_dataset(file_pattern: str) -> tf.data.Dataset:
    """
    Create a dataset from files matching the provided pattern.

    :param file_pattern: File pattern
    :return: A dataset
    """

    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)
    dataset = dataset.prefetch(tf.data.AUTOTUNE).map(process_image, tf.data.AUTOTUNE).cache()

    return dataset


def measure_performance(dataset: tf.data.Dataset) -> None:
    """
    Measure how long it takes to iterate over the entire dataset.

    :param dataset: A dataset
    """

    start_time = time.perf_counter()

    for _ in dataset:
        time.sleep(0.01)

    stop_time = time.perf_counter()

    print(f"execution time: {stop_time - start_time}")


def main():

    # create dataset
    dataset = tf.data.Dataset.list_files('data/chicago-face-database/*', shuffle=False)

    # using sequential mapping -> 18.491415916
    # dataset = dataset.map(process_image)
    # measure_performance(dataset)

    # using parallel mapping -> 18.477471
    # dataset = dataset.map(process_image, tf.data.AUTOTUNE)
    # measure_performance(dataset)

    # using caching -> 18.300266209
    # dataset = dataset.map(process_image).cache()
    # measure_performance(dataset)

    # using prefetch -> 18.234782334
    # dataset = dataset.prefetch(tf.data.AUTOTUNE).map(process_image).cache()
    # measure_performance(dataset)

    # using parallel mapping, caching and prefetch -> 18.232624416
    dataset = dataset.prefetch(tf.data.AUTOTUNE).map(process_image, tf.data.AUTOTUNE).cache()
    measure_performance(dataset)


if __name__ == '__main__':
    main()
