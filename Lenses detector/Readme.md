# Lenses Detector using OpenCV

This project implements a real-time eyelenses detection system using OpenCV and the Haar Cascade Classifier. It detects eyelenses in video frames captured from a webcam and displays the detection with bounding boxes. Additionally, it logs the detection results (timestamps and coordinates) into an HTML file, which can be viewed later to analyze the detection results.

## Features
**Eyeglass Detection**: Utilizes the pre-trained Haar Cascade Classifier for detecting eyeglasses in real-time.
**Webcam Integration**: Captures video feed from the webcam and processes each frame for detection.
**Bounding Boxes**: Displays detected eyeglasses by drawing bounding boxes around them in the video feed.
**Result Logging**: Logs detection results (timestamp and coordinates) into a results.html file for review.
**Easy Visualization**: Open the results.html file in a browser to view a table of all the detection events.

## Requirements
Python 3.x
OpenCV (cv2)
A webcam 
