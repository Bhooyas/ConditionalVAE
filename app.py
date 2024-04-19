import streamlit as st
from inference import *
import math

def generate_num(ip_str):
    if ip_str == "":
        return
    
    num = get_num(ip_str)
    st.write(f"Detected number {num}")

    if num is None:
        return
    
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
                return

st.title('Number Generator using Conditional VAE')

ip_str = st.text_input("Enter an number", key="text")
times = st.slider("Select number of images to generate", 9, 225, value=100)

if st.button('Generate Images'):
    generate_num(ip_str)

if st.button("Generate Sample Image"):
    ip_str = get_sample()
    st.write(f"Input Sentence: - {ip_str}")
    generate_num(ip_str)