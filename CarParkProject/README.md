# 🎥 Real-Time-Car-Parking-Detection

This project aims to develop a real-time parking space detection system using deep learning techniques, specifically the U-Net architecture for accurate image segmentation. The system identifies available parking spots from live video feeds, enhancing urban mobility.

## 📖 Table of Contents
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Complete Process Guide](#complete-process-guide)
- [Sample Example](#sample-example)
- [Acknowledgments](#acknowledgments)

## Features

- **U-Net Architecture:**: Enables precise pixel-wise segmentation to differentiate between occupied and free parking spaces.
- **Real-Time Processing:**: Provides immediate feedback for drivers, crucial in busy urban environments.
- **Applications**: Useful for smart city initiatives, autonomous vehicles, and traffic management.

## Requirements

This project requires the following Python libraries:
- `opencv-python` for video processing.
- `numpy` for numerical operations.
- `pickle` to display dynamic content on screens based on real-time data inputs.
- `cvzone` simplifies the implementation of image processing and AI functions
- Any other dependencies specified in the notebook.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jaidh01/Computer-Vision-Projects.git
   cd Computer-Vision-Projects/CarParkProject
   ```

2. Install the required dependencies:

## Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook main.ipynb
   ```

2. Follow the instructions in the notebook to record/upload your video file and adjust any parameters as needed.

3. Run the cells sequentially to process the video and train the ML model.

## Complete Process Guide

1. **Video Input**: Begin by uploading the video feed from the parking area into the designated section of the notebook.
  
2. **Frame Extraction**: The system will automatically extract frames from the uploaded video, enabling you to analyze individual frames for parking space detection.

3. **Labelling Extracted Data**: Designate specific keys to label your data. As the video plays, press the assigned key to mark occupied or free parking spaces in each frame.

4. **Data Preparation**: After labeling, you will generate a .csv file containing the labeled data, which can be split into training and testing sets for model training.

5. **Model Selection**: You can choose from various machine learning models (e.g., Decision Trees, Random Forests, etc.) according to your need.

6. **Training**: Train the selected model using the prepared dataset.

7. **Evaluation**: After training, evaluate the model's performance using metrics such as Intersection over Union (IoU), accuracy, and F1 score to assess its effectiveness in detecting parking spaces.

8. **Visualization**:  Finally, visualize the segmentation results on sample frames to understand how well the model identifies available and occupied parking spaces in real-time.

## Sample Example


## Acknowledgments

- Special thanks to the open-source community for the libraries and tools that made this project possible.
- [OpenCV](https://opencv.org/) for its powerful video processing capabilities.
- [Scikit-learn](https://scikit-learn.org/stable/) for machine learning functionalities.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For any inquiries or issues, please feel free to open an issue in the repository or contact me at [jaidhinrgra402@gmail.com].
