# Copyright (c) 2022 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# -----> streamlit run app.py

import streamlit as st
import numpy as np
import tensorflow as tf


# location of the pretrained model

MODEL_LOC = 'data/results/2022-12-29-09-53-16/decoder'

# [min_value, max_value, value, step, disabled] of each slider

VALS = [
    [-2.64, 2.65, 0.0, 0.01, False],
    [-2.77, 2.77, 0.0, 0.01, False],
    [-2.80, 2.65, 0.0, 0.01, False],
    [-0.03, 0.02, 0.0, 0.01, True],  # disable
    [-2.84, 2.77, 0.0, 0.01, False],
    [-2.71, 2.60, 0.0, 0.01, False],
    [-2.83, 2.72, 0.0, 0.01, False],
    [-2.40, 2.45, 0.0, 0.01, False],
    [-0.02, 0.06, 0.0, 0.01, True],  # disable
    [-2.59, 2.59, 0.0, 0.01, False]
]


def generate(code):
    """ Generates image from latent code """
    img = generator(np.expand_dims(np.array(code), axis=0))
    img = np.uint8(img[0] * 255)
    return img


# add title and subtitle

st.title('Image Generator')
st.subheader('Adjust the sliders to modify the appearance')

# create layout

left_col, _, mid_col, _, right_col = st.columns([2, 1, 6, 1, 2])

# create sliders

code = []  # array used to store values of sliders

for i in range(0, 5):
    code.append(left_col.slider(
        label=f'DIM {i}',
        min_value=VALS[i][0],
        max_value=VALS[i][1],
        value=VALS[i][2],
        step=VALS[i][3],
        label_visibility='visible',
        format=None,
        disabled=VALS[i][4],
    ))

for i in range(5, 10):
    code.append(right_col.slider(
        label=f'DIM {i}',
        min_value=VALS[i][0],
        max_value=VALS[i][1],
        value=VALS[i][2],
        step=VALS[i][3],
        label_visibility='visible',
        format=None,
        disabled=VALS[i][4],
    ))

# load generator model

generator = tf.keras.models.load_model(MODEL_LOC)

# generate image

image = generate(code)
mid_col.image(
    image=image,
    use_column_width="always"
)


