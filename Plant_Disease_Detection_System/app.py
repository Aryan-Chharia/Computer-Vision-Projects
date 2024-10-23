import os
import time
import cv2
import sys
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import keras_preprocessing
import streamlit as st
import numpy as np
import pandas as pd
from numpy import argmax
from PIL import Image, ImageEnhance
from resizeimage import resizeimage
from utils import label_map_util
from utils import visualization_utils as vis_util
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from pathlib import Path

# Ensure compatibility with TensorFlow 2.x
tf.executing_eagerly()

MODEL_NAME = './object_detection/inference_graph'
IMAGE_NAME = './object_detection/images/out.jpg'

CWD_PATH = os.getcwd()

PATH_TO_CKPT = os.path.join('./object_detection/inference_graph/frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('./object_detection/training/labelmap.pbtxt')
PATH_TO_IMAGE = os.path.join('./object_detection/images/out.jpg')

NUM_CLASSES = 6

# Load label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load frozen TensorFlow model into memory
detection_graph = tf.compat.v1.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.compat.v1.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Load image using OpenCV
in_image = cv2.imread(PATH_TO_IMAGE)
image_rgb = cv2.cvtColor(in_image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# Perform the actual detection by running the model
(boxes, scores, classes, num) = sess.run(
    [detection_boxes, detection_scores, detection_classes, num_detections],
    feed_dict={image_tensor: image_expanded})

# Draw the results of the detection
# Updated fix for FreeTypeFont issue using getbbox()
vis_util.visualize_boxes_and_labels_on_image_array(
    in_image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.60)

# Update getsize() to getbbox() inside visualization_utils.py:
# Replace this line:
# display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
# With this:
# display_str_heights = [font.getbbox(ds)[3] - font.getbbox(ds)[1] for ds in display_str_list]


# Streamlit UI components
def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

def classification():
    global path
    global cn
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(24, activation='softmax')
    ])

    LABELS = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry___healthy', 'Cherry___Powdery_mildew', 'Grape___Black_rot',
        'Grape___Esca_Black_Measles', 'Grape___healthy', 'Grape___Leaf_blight_Isariopsis_Leaf_Spot',
        'Orange___Haunglongbing', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper_bell___Bacterial_spot', 'Pepper_bell___healthy', 'Potato___Early_blight',
        'Potato___healthy', 'Potato___Late_blight', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___healthy', 'Strawberry___Leaf_scorch'
    ]

    model.load_weights("./object_classification/rps.h5")
    path = './object_classification/images/out.jpg'
    img = image.load_img(path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    result = argmax(classes)
    cn = LABELS[result]

classification()

# Main Streamlit app function
def main():
    st.markdown('<style>body{-webkit-app-region: drag;}</style>', unsafe_allow_html=True)
    st.title("Plant Disease Detection & Classification")
    st.text("Built with Streamlit and TensorFlow")
    
    activities = ["About", "Plant Disease"]
    choice = st.sidebar.selectbox("Select Activity", activities)
    enhance_type = st.sidebar.radio("Type", ["Detection", "Classification", "Treatment"])
    
    if choice == 'About':
        intro_markdown = read_markdown_file("./doc/about.md")
        st.markdown(intro_markdown, unsafe_allow_html=True)

    if choice == 'Plant Disease' and enhance_type == 'Detection':
        st.header("Plant Disease Detection")
        image_file = st.file_uploader("Upload Image", type=['jpg'])
        st.markdown("* * *")
        
        if image_file is not None:
            our_image = Image.open(image_file)
            our_image.save('./object_detection/images/out.jpg')
            
            if st.button('Process'):
                st.image(in_image, use_column_width=True, channels='RGB')
            st.image(our_image, use_column_width=True, channels='RGB')
            st.balloons()

    if choice == 'Plant Disease' and enhance_type == 'Classification':
        st.header("Plant Disease Classification")
        image_input = st.file_uploader("Upload Image", type=['jpg'])
        st.markdown("* * *")
        
        if image_input is not None:
            some_image = Image.open(image_input)
            some_image.save('./object_classification/images/out.jpg')
            
            if st.button('Classify'):
                st.image(path, use_column_width=True)
                with st.spinner('Your image is processing'):
                    time.sleep(5)
                st.success(cn)
                st.balloons()

    if enhance_type == 'Treatment' and choice == 'Plant Disease':
        data_markdown = read_markdown_file("./treatment/treatment.md")
        st.markdown(data_markdown, unsafe_allow_html=True)

main()
