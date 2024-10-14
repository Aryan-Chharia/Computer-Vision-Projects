# Neck Posture Monitoring Using OpenCV and Mediapipe

This project implements a real-time neck posture monitoring system using OpenCV and Mediapipe. The system detects the user's face and neck, calculates the neck angle, and provides posture warnings when the user is looking down for extended periods, which can lead to "text neck" and other posture-related issues.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Results](#results)
- [Troubleshooting](#troubleshooting)

### Requirements
To run this project, you will need:
- Python 3.7+
- OpenCV 4.5+
- Mediapipe
- Numpy

### Installation

Clone the repository:

```bash
git clone https://github.com/your-repo/neck-posture-monitoring.git
cd neck-posture-monitoring
```

### Install dependencies:

Install the required Python packages using pip:

```bash
pip install numpy opencv-python mediapipe
```
### Usage
#### Prepare the environment:
Ensure that your camera is connected and working.

### Run the script:
To run the neck posture monitoring script, use the following command:

```bash
python neck_posture_monitoring.py
```
The script will detect the user's face and neck, calculate the neck angle, and display posture warnings if the user maintains a poor posture for too long.

### Project Structure

```bash
.
├── neck_posture_monitoring.py        # Main script to run neck posture detection
├── README.md                         # This file

```

### Models

This project uses Mediapipe for pose detection, which does not require external model files. Mediapipe automatically detects body landmarks, including the neck and shoulders, to calculate the neck angle.

### Results

The system will output a real-time video stream with:

- Detected Neck Angle: The system calculates the angle between the neck and shoulders.
- Posture Warning: If the neck angle indicates poor posture (e.g., looking down for too long), a warning will appear on the screen.


### Expected Output:

- Real-time detection of neck posture.
- Warnings displayed when poor posture is detected.


### Troubleshooting

#### 1. Camera not detected:
Ensure your camera is connected and functioning properly. Use a camera testing tool or check the device in OpenCV with:

```bash
cap = cv2.VideoCapture(0)
```

#### 2. Incorrect neck angle or no warning:
Ensure you are in a well-lit environment for better detection of face and neck. Adjust the threshold for the neck angle if needed.

#### 3. Slow frame rate:
Lower the frame resolution or increase the confidence thresholds for Mediapipe to improve performance.

#### 4. Mediapipe errors:
Ensure that you have the latest version of Mediapipe installed and all necessary packages are up-to-date.
