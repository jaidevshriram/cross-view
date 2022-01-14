import streamlit as st
import cv2
import os

from PIL import Image

SCENE_PATH = "/scratch/jaidev/HabitatGibson/data/"
BEV_PATH = "/scratch/jaidev/HabitatGibson/bevs/sim/"
NUM_ITEMS = 3
COLUMNS = 3
NUM_COLS = NUM_ITEMS * COLUMNS


scenes = os.listdir(SCENE_PATH)

scene = st.sidebar.selectbox(
        "Select the scene",
        scenes
        )

st.title("Scene Name: " + scene)

img_count = len(os.listdir(os.path.join(SCENE_PATH, scene, "0", "left_rgb")))
idx = 0

@st.cache(allow_output_mutation=True)
def load_img(path):
    img = Image.open(path)
    return img

while idx < img_count:

    st_row = st.container()
    cols = st_row.columns(NUM_COLS)

    for icol in range(NUM_ITEMS):

        if idx >= img_count:
            break

        rgb_img = f"{SCENE_PATH}/{scene}/0/left_rgb/{idx}.jpg"
        bev_img = f"{BEV_PATH}/{scene}/0/bev/{idx}.png"


        cols[COLUMNS * icol].image(load_img(rgb_img), caption=f"{idx}")
        cols[COLUMNS * icol + 1].image(load_img(bev_img))
        agree = cols[COLUMNS * icol + 2].checkbox('', key=idx)

        if agree:
            f = open(f'{idx}.txt', 'w')
            f.close()

        idx += 1


