# Copyright (c) 2022 Viktor Fairuschin
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# -----> streamlit run app.py

import streamlit as st
import numpy as np
import tensorflow as tf


# location of the pretrained model

MODEL_LOC = 'data/results/pretrained/decoder'

# [min_value, max_value, value and step] of each slider

VALS = [
    [-1.0, 1.0, 0.0, 0.1],
    [-1.0, 1.0, 0.0, 0.1],
    [-1.0, 1.0, 0.0, 0.1],
    [-1.0, 1.0, 0.0, 0.1],
    [-1.0, 1.0, 0.0, 0.1],
    [-1.0, 1.0, 0.0, 0.1],
    [-1.0, 1.0, 0.0, 0.1],
    [-1.0, 1.0, 0.0, 0.1],
    [-1.0, 1.0, 0.0, 0.1],
    [-1.0, 1.0, 0.0, 0.1]
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
    ))

# load generator model

generator = tf.keras.models.load_model(MODEL_LOC)

# generate image

image = generate(code)
mid_col.image(
    image=image,
    use_column_width="always"
)


