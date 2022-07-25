import streamlit as st
import pandas as pd
import numpy as np
import torch
# import opencv-python-headless as cv2
import cv2
from tempfile import NamedTemporaryFile
# import os

@st.cache
def load_model(project_path, model_path):
    return torch.hub.load(project_path, 'custom', path=model_path, source='local')

# @st.cache
def draw_rectangle(image, detection):
    xmin, ymin, xmax, ymax, conf, classification, name = detection
    color = (255, 0, 0) if classification == 0 else (0, 0, 255)
    pos = (int(xmin)+5, int(ymax)-5)
    cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
    cv2.putText(image, f'{name} {conf:.2f}', pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def process_image(image):
    result = model(image)
    result.pandas().xyxy[0]
    for detection in result.pandas().xyxy[0].to_numpy():
        draw_rectangle(image, detection)

    st.image(image)

def process_video(file):

    cap = cv2.VideoCapture(file)
    stframe = st.empty()
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = model(frame)
            for detection in result.pandas().xyxy[0].to_numpy():
                draw_rectangle(frame, detection)
            stframe.image(frame)
        else:
            break
    cap.release()



# -------------------------------------------- WebApp -----------------------------------------------------

st.title("Détection d'incendies urbaines utilisant YOLOv5m")

# st.text("Le dataset est composé par 325 images, dont 23 (7%) sont des images du background")

yolo_path = './'
model_path = yolo_path + '/medium/fine_tuning/weights/best.pt'
model = load_model(yolo_path, model_path)

uploaded_file = st.file_uploader('Déposez une vidéo ou une image pour apliquer le modèle', type=['jpg', 'png', 'jpeg', 'mp4'])
if uploaded_file is not None:
    ext = uploaded_file.name.split('.')[1]
    if ext == 'mp4':
        tfile = NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        _, frame = cv2.VideoCapture(tfile.name).read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(frame)
        if st.button('Detection'):
            process_video(tfile.name)
    else:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image)
        if st.button('Detection'):
            process_image(image)
