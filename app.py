import streamlit as st
from inference import infer
import math

st.title('Number Generator using Conditional VAE')

num = st.slider("Select a number to generate", 0, 9)
times = st.slider("Select number of images to generate", 9, 225, value=100)

if st.button('Generate Images'):
    images = infer(num, times)

    grid = math.ceil(math.sqrt(times))

    st.subheader('Generated Images')
    for i in range(grid):
        cols = st.columns(grid)
        for j in range(len(cols)):
            index = (i * grid) + j
            if index < len(images):
                cols[j].image(images[index].squeeze(), use_column_width=True)
            else:
                break