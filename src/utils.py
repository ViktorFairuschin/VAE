import tensorflow as tf


def parse_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(image, channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, size=[184, 128], preserve_aspect_ratio=False)
    return image


def create_dataset(path):
    return tf.data.Dataset.list_files(path).map(parse_image)


def create_train_dataset(path, batch_size=16):

    def _preprocess(image):
        return image, image

    ds = create_dataset(path)
    ds = ds.shuffle(2000).batch(batch_size).map(_preprocess).cache().prefetch(tf.data.AUTOTUNE)
    return ds

