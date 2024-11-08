# Virtual Keyboard with Hand Gesture Control 

## Project Overview
This project implements a virtual keyboard that can be controlled using hand gestures captured through a webcam. Users can type by making pinching gestures in the air, making typing possible without physical contact with any surface.

## Features
- ✋ Hand gesture-controlled virtual keyboard
- 👆 Pinch-to-type functionality
- ✊ Fist gesture to close keyboard
- ⌫ Backspace functionality
- 🎯 Reduced click sensitivity for accurate typing
- 💡 Visual feedback with lighting effects
- 🎨 Enhanced UI with centered keyboard layout
- 📝 Real-time text display

## Requirements
- Python 3.7+
- OpenCV
- CVZone
- Mediapipe
- NumPy
- PyAutoGUI

For exact versions, please refer to the `requirements.txt` file.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Jaival111/Gesture-Based-Virtual-Keyboard.git
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the main script:
```bash
python virtual_keyboard.py
```

2. Gesture Controls:
   - Move your hand to hover over virtual keys
   - Pinch index finger and thumb to "press" keys
   - Make a fist to close the keyboard
   - Use the "BACK" button for backspace
   - Use the "CLOSE" button or make a fist to exit

## Technical Details

### Hand Detection
- Uses CVZone's HandTrackingModule
- Detection confidence threshold: 0.8
- Supports tracking of up to 2 hands

### Keyboard Layout
- QWERTY layout with special characters
- Additional control buttons (Backspace, Space, Close)
- Enhanced visual feedback system

### Performance Optimizations
- Implemented click delay to prevent multiple inputs
- Optimized gesture recognition thresholds
- Improved UI positioning for better usability

## Known Issues
1. May require good lighting conditions for optimal hand detection
2. Performance depends on webcam quality
3. Might need calibration for different hand sizes

## Future Improvements
- [ ] Add support for special characters
- [ ] Implement predictive text
- [ ] Add customizable keyboard layouts
- [ ] Improve gesture recognition in low light
- [ ] Add support for different languages
