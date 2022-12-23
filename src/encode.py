# Copyright (c) 2022 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf


decoder = tf.keras.models.load_model('data/results/2022-12-23-15-36-34/decoder')
decoded = decoder(np.zeros((1, 10)))

fig, ax = plt.subplots()
ax.imshow(decoded[0],  cmap='gray')
ax.set_xticks([])
ax.set_yticks([])
plt.show()

