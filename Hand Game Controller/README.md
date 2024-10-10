# Hand Controller: Control Cars 🚗 Using Hand Gestures in Games

This project leverages Python and OpenCV to create a gesture-based hand controller for games. Using computer vision techniques, the program tracks the user's hand movements and translates them into real-time actions to control a car in any game. This unique and intuitive interface allows players to steer, accelerate, brake, and perform other functions without the need for physical controllers or keyboards.

## Features

- **Real-time Hand Tracking:** Uses OpenCV to detect hand gestures.
- **Gesture-based Controls:** Move your hand to steer the car, accelerate, or change directions.
- **Supports PC and Mobile Cameras:** Control the car using either your PC's webcam or your mobile device's camera.
- **Cross-Game Compatibility:** Can be used to control vehicles in various driving games by mapping hand gestures to game actions.

## How It Works

This project captures the live feed from a camera (either from your PC or mobile) and detects specific hand gestures. These gestures are then translated into game commands (e.g., moving left, right, accelerating, or braking). The gesture detection works using basic image processing techniques, including contour detection and hand tracking algorithms.

### Key Gestures:
- **Open hand** – Move the car forward
- **Closed fist** – Stop the car
- **Move hand left** – Steer left
- **Move hand right** – Steer right

These gestures are customizable and can be adjusted based on user preferences or the specific game being played.

## Installation

To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/hand-controller.git
cd hand-controller
pip install -r requirements.txt
```

Make sure you have Python 3.x installed along with the required libraries in `requirements.txt`. 

### Requirements:

- Python 3.x
- OpenCV
- NumPy

## Usage

You can use either your PC's camera or your mobile camera to control the car. 

### Using PC Camera:
1. Run the following command to start the program with your PC webcam:
    ```bash
    python main-pc-cam.py
    ```
2. Put your hand in front of the camera, and the program will start detecting gestures.

### Using Mobile Camera:
1. Download the [IP Webcam](https://play.google.com/store/apps/details?id=com.pas.webcam) app on your mobile device.
2. Launch the app and note the IP address shown on the screen.
3. In the `main-mobile-cam.py` script, update the `url` variable with the IP address provided by the IP Webcam app.
4. Run the following command to start the program using your mobile camera:
    ```bash
    python main-mobile-cam.py
    ```
5. Position your hand in front of the mobile camera to control the car.

### Customizing Gesture Control:
You can modify the gesture mappings in the main script files (`main-pc-cam.py` or `main-mobile-cam.py`) to assign specific hand gestures to different game actions. Feel free to experiment with different gestures to fit your playstyle.
