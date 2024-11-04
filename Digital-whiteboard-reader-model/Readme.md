# Digital Whiteboard Reader

This project is a digital whiteboard reader model using computer vision. It captures text and sketches from a whiteboard in real time, converting them into digital notes that can be saved, organized, and shared. This tool is useful for lectures, meetings, or brainstorming sessions.

## Features

- **Real-time Capture**: Captures text and sketches from a whiteboard in real-time.
- **Text Recognition**: Uses Tesseract OCR to extract text from the whiteboard content.
- **Bounding Boxes**: Highlights detected text with bounding boxes for easier readability.
- **Digital Note Creation**: Allows easy conversion of handwritten notes into digital format.

## Requirements

## To run this project, install the following Python libraries:

# Installation Instructions

To set up the project, follow these steps:

1. Install Required Libraries:
   ```bash
   pip install opencv-python pytesseract numpy

2. You also need to install Tesseract OCR.

# Setup

1. Clone or Download this Repository

2. Install Required Libraries:
  -- pip install opencv-python pytesseract numpy

3. Update the Tesseract path in main.py:
  -- pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

## Run the application

-- python main.py

## Code Explanation
-- preprocess_image: Converts the image to grayscale, applies blur, and thresholding to enhance OCR accuracy.
-- extract_text: Extracts text from the processed image using Tesseract OCR.
-- extract_text_with_boxes: Draws bounding boxes around detected words.
-- capture_whiteboard: Captures and processes frames from the webcam in real-time.
