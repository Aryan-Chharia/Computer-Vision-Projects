# data_preprocessing.py

import cv2
import os

def preprocess_image(image_path, output_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 128))
    cv2.imwrite(output_path, image)

if __name__ == "__main__":
    input_folder = "data/raw/"
    output_folder = "data/processed/"
    os.makedirs(output_folder, exist_ok=True)
    for image_file in os.listdir(input_folder):
        preprocess_image(os.path.join(input_folder, image_file), os.path.join(output_folder, image_file))
