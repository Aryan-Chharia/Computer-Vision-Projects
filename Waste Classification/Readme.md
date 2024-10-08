# Waste Classification Project

## Overview
This project implements a real-time waste classification system using a webcam and a pre-trained Keras model. It uses OpenCV for image processing and overlays appropriate images of detected waste and bins onto a background to guide the user on the correct waste disposal bin.

## Features
- **Real-Time Waste Classification**: Classifies waste into four categories: Recyclable, Hazardous, Food, and Residual.
- **Visual Feedback**: Displays detected waste and highlights the correct bin using arrows and waste images.
- **Simple User Interface**: Provides a clear display with a webcam feed, detected waste image, and the correct bin for disposal.
- **Interactive**: Automatically classifies waste in real-time through the webcam feed.

## Project Structure   
Waste-Classification-Project/
├── main.py                     
├── README.md                  
├── Resources/                  
│   ├── arrow.png              
│   ├── bgimg.png               
│   ├── Model/                 
│   │   ├── keras_model.h5      
│   │   └── labels.txt          
│   ├── Waste/                  
│   │   ├── 1.png         
│   │   ├── 2.png          
│   │   └── ...                 
│   └── Bins/                   
│       ├── 1.png            
│       ├── 2.png            
│       └── ...    

## Requirements
- Python 3.x
- OpenCV (`cv2`)
- cvzone (`pip install cvzone`)
- TensorFlow (`pip install tensorflow`)