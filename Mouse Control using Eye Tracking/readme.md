# Eye-Controlled Mouse using Eye Tracking

This project uses a webcam to track eye movements and control the mouse cursor on your screen. It utilizes the **MediaPipe Face Mesh** for real-time facial landmark detection and **PyAutoGUI** to move the mouse based on eye landmarks. Additionally, it can simulate a mouse click when a blink is detected.

## Features
- Tracks eye movements and moves the mouse pointer accordingly.
- Simulates mouse clicks by detecting eye blinks.
- Real-time video display showing eye landmarks.

## Technologies Used
- **OpenCV**: For capturing video from the webcam and drawing landmarks on the video frame.
- **MediaPipe**: For facial landmark detection, specifically focusing on eye landmarks.
- **PyAutoGUI**: For controlling the mouse movements and clicks based on eye tracking.

## Prerequisites

Before running the script, ensure that you have the required libraries installed.

### Installation

1. **Python 3.x** should be installed on your machine.
2. Install the necessary Python libraries by running the following command:

```bash
pip install opencv-python mediapipe pyautogui
```

## How it Works

1. **Face Mesh Detection**:
   - MediaPipe's **Face Mesh** model is used to detect and track facial landmarks in real-time.
   - The eye landmarks (landmarks 474 to 478) are mapped to your screen's dimensions, enabling mouse control.
  
2. **Mouse Movement**:
   - The eye's position on the camera feed is converted into screen coordinates, and the mouse pointer is moved accordingly using **PyAutoGUI**.

3. **Blink Detection for Click**:
   - The script monitors the distance between two specific landmarks on the left eye (landmarks 145 and 159).
   - If the distance between these points is less than a specified threshold, a blink is detected, triggering a mouse click.

## Usage

1. **Run the Script**:
   - Use the following command to run the Python script:

   ```bash
   python main.py
   ```

2. **Control the Mouse**:
   - Move your eyes, and the mouse cursor will follow.
   - Blink to simulate a mouse click.

3. **Exit**:
   - Press the `q` key to quit the application.

## Troubleshooting

1. **Mouse Not Moving**: Ensure that you have given your Terminal or IDE the necessary permissions to control the mouse.
   - Go to **System Preferences** > **Security & Privacy** > **Privacy** > **Accessibility** and allow your terminal/IDE access.

2. **Accuracy of Movement**: If the mouse movement isn't smooth or precise, try adjusting the coordinates mapping or landmarks used for tracking.

3. **Blink Detection**: The blink threshold might need to be adjusted depending on the lighting conditions and distance from the camera.


