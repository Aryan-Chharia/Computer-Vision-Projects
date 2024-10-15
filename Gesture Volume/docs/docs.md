# Hand Gesture-Based Volume Control

## Working Principle

The Gesture-based **Volume Control Tool** is an innovative application that allows users to adjust their computer's volume using intuitive hand gestures. By leveraging computer vision and machine learning techniques, the tool interprets specific hand movements captured through a webcam to control system audio levels. This creates a touchless, intuitive interface for volume adjustment.

## Core Components

### Main Tasks
1. Hand Detection
2. Gesture Analysis
3. Volume Adjustment

## Detailed Workflow

### 1. Hand Detection

The system employs the **Mediapipe library** for robust hand tracking:
- Captures real-time video feed from the computer's webcam
- Processes each frame to identify hand presence
- Extracts key landmarks, particularly focusing on the thumb and index finger
- Calculates the distance between these fingers for gesture interpretation


### 2. Gesture Analysis

The tool processes hand positioning to interpret gestures:
- Measures the distance between thumb and index finger
- Normalizes this distance to a scale of 0 to 100
- Maps the normalized distance to volume levels
- Implements smoothing to prevent erratic volume changes

#### Distance Calculation
```python
def calculate_distance(self, p1, p2):
    """Calculate Euclidean distance between two points"""
    return int(hypot(p2[0] - p1[0], p2[1] - p1[1]))
```

### 3. Volume Adjustment

The system translates analyzed gestures into volume controls:
- Uses **PyCaw** library to interface with system audio
- Maps normalized hand distances to volume levels:
  - Minimum distance (fingers together) = Minimum volume
  - Maximum distance (fingers apart) = Maximum volume
- Provides visual feedback of current volume level

#### Volume Control Implementation
```python
            volBar = np.interp(length, [50, 200], [400, 150])
            volPer = np.interp(length, [50, 200], [0, 100])

            # Reduce Resolution to make it smoother
            smoothness = 5
            volPer = smoothness * round(volPer / smoothness)

            # Check fingers up
            fingers = detector.fingersUp()
            # print(fingers)

            # If pinky is down set volume
            if not fingers[4]:
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)
```

## User Interface

The tool provides a real-time visual interface:
- Displays webcam feed with hand tracking visualization
- Shows volume level as a percentage
- Provides a visual bar indicating current volume
- Highlights detected hand landmarks for user feedback

## Technical Implementation

### Libraries Used
- **OpenCV**: Video capture and image processing
- **Mediapipe**: Hand detection and landmark tracking
- **PyCaw**: Windows audio control interface
- **NumPy**: Numerical operations and data processing

### System Requirements
- Windows operating system
- Webcam
- Python 3.7 or higher
- Required libraries (see `requirements.txt`)

## Installation Guide

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gesture-volume-control.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python volume_control.py
```

## Usage Instructions

1. Launch the application
2. Position your hand in view of the webcam
3. Adjust volume by changing the distance between thumb and index finger:
   - Bring fingers together to lower volume
   - Move fingers apart to increase volume
4. Press 'q' to exit the application

## Future Enhancements

- **Multi-gesture support**: Implement additional gestures for other audio controls
- **Customizable gestures**: Allow users to define their own gesture mappings
- **Enhanced stability**: Implement advanced filtering for more stable volume control
- **Cross-platform support**: Extend functionality to macOS and Linux

## Troubleshooting

Common issues and solutions:
1. **Hand not detected**: Ensure adequate lighting and clear background
2. **Erratic volume changes**: Adjust sensitivity in settings
3. **Performance issues**: Check system requirements and close resource-intensive applications

## Contributing

We welcome contributions! Please follow these steps:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the Apache License - see the [LICENSE](LICENSE) file for details.

---

## Contact

For support or queries:
- Email: mjgandhi2305@gmail.com
- GitHub Issues: [Project Issues Page](https://github.com/yourusername/gesture-volume-control/issues)
