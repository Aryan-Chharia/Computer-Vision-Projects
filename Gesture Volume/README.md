# Gesture Volume Control

![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-informational)
[![Python](https://img.shields.io/badge/Python-%203.8%20%7C%203.9%20%7C%203.10-informational)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache-green)](./LICENSE)

Control your system volume with hand gestures using computer vision and hand tracking technologies.

## Overview

This project provides a hand-tracking application that allows users to control system volume through hand gestures. It utilizes OpenCV and MediaPipe for hand detection and tracking, and includes a Tkinter GUI for interaction. The application works on Windows, macOS, and Linux.

## Features

- **Hand Tracking:** Uses MediaPipe to detect hand movements in real-time.
- **Volume Control:** Adjusts system volume based on hand gestures, specifically the distance between the thumb and index finger.
- **User Interface:** A Tkinter GUI that displays the current volume and allows users to start/stop tracking.
- **Settings:** Options for camera selection and gesture sensitivity adjustments.
- **Logging:** Logs application activities and errors for debugging purposes.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- PyAutoGUI
- NumPy
- Tkinter

## Installation

1. Clone this repository:
   ``` bash
      git clone https://github.com/Aryan-Chharia/Computer-Vision-Projects
      cd /Computer-Vision-Projects/Gesture Volume
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. #### For Linux users, ensure that `amixer` is installed for volume control. Use the package manager to install it if necessary.

## Usage

1. Run the application:
   ```bash
   python Gesture-Volume.py
   ```

2. In the GUI:
   - Select your camera if you have multiple devices.
   - Adjust the sensitivity slider to modify how responsive the volume control is to hand movements.
   - Click "Start Tracking" to begin hand gesture recognition.
   - Use the thumb and index finger to adjust the volume:
     - Move them apart to increase volume.
     - Bring them closer together to decrease volume.

3. Click "Stop Tracking" to halt the tracking process.
4. Show your Palm to select volume

## Code Structure

### HandTrackingModule.py

- **HandDetector:** Class responsible for detecting and tracking hands using MediaPipe.
- **Application:** Tkinter GUI application that initiates hand tracking and updates the cursor position based on hand movements.

### GesVol.py

- **VolumeController:** Class that interacts with the system to adjust the volume using platform-specific commands.
- **VolumeControlApp:** Main application class that manages the GUI and integrates the hand tracking functionality.

## Logging

The application logs information and errors to `volume_control.log`. This can be useful for debugging and understanding application performance.

## Contributing

Contributions are welcome! If you would like to add features or fix bugs, feel free to submit a pull request.

## Acknowledgments

- [OpenCV](https://opencv.org/) for computer vision capabilities.
- [MediaPipe](https://google.github.io/mediapipe/) for hand tracking.
- [Tkinter](https://docs.python.org/3/library/tkinter.html) for GUI development.


---
## Developed By

This application was developed by [Dawood](htttps://github.com/Dwukn). 

## Contact

For any inquiries, suggestions, or issues, please contact us via:

- Email: dawood220a@gmail.com
- Linkden: [Dawood](https://www.linkedin.com/in/dwukn/)

Thank you for using our application!
