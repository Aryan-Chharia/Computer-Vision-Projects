import cv2
import time
import numpy as np
import HandTrackingModule as htm  # Ensure this module is correctly implemented
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

################################
wCam, hCam = 640, 480  # Set the width and height of the webcam feed
################################

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(3, wCam)  # Set the width
cap.set(4, hCam)  # Set the height
pTime = 0  # Previous time for FPS calculation

# Initialize hand detector
detector = htm.handDetector(detectionCon=0.7, maxHands=1)

# Set up volume control using Pycaw
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()  # Get the volume range
minVol = volRange[0]  # Minimum volume
maxVol = volRange[1]  # Maximum volume
vol = 0
volBar = 400  # Initial volume bar height
volPer = 0  # Volume percentage
area = 0  # Area for hand detection
colorVol = (255, 0, 0)  # Color for volume display

while True:
    success, img = cap.read()  # Capture the frame
    if not success:
        print("Error: Could not read frame from webcam.")
        break

    # Find Hand
    img = detector.findHands(img)  # Detect hands in the image
    lmList, bbox = detector.findPosition(img, draw=True)  # Get landmarks and bounding box
    if len(lmList) != 0:  # If hand is detected
        # Filter based on size of bounding box
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100  # Calculate the area of the bounding box
        if 250 < area < 1000:  # Only consider hands of a certain size
            # Find Distance between index finger and thumb
            length, img, lineInfo = detector.findDistance(4, 8, img)

            # Convert the distance to volume
            volBar = np.interp(length, [50, 200], [400, 150])  # Map distance to volume bar height
            volPer = np.interp(length, [50, 200], [0, 100])  # Map distance to volume percentage

            # Reduce Resolution to make it smoother
            smoothness = 5
            volPer = smoothness * round(volPer / smoothness)  # Smooth the volume percentage

            # Check fingers up
            fingers = detector.fingersUp()  # Get the state of fingers

            # If pinky is down, set the volume
            if not fingers[4]:  # If pinky is down
                volume.SetMasterVolumeLevelScalar(volPer / 100, None)  # Set the master volume
                if lineInfo:  # Check if lineInfo has values
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)  # Draw a circle
                colorVol = (0, 255, 0)  # Change color to green
            else:
                colorVol = (255, 0, 0)  # Reset color to red if pinky is up

    # Drawings
    cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)  # Draw the volume bar outline
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)  # Fill the volume bar
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)  # Display volume percentage
    cVol = int(volume.GetMasterVolumeLevelScalar() * 100)  # Get current volume level
    cv2.putText(img, f'Vol Set: {int(cVol)}', (400, 50), cv2.FONT_HERSHEY_COMPLEX, 1, colorVol, 3)  # Display set volume

    # Frame rate calculation
    cTime = time.time()  # Current time
    fps = 1 / (cTime - pTime)  # Calculate FPS
    pTime = cTime  # Update previous time
    cv2.putText(img, f'FPS: {int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)  # Display FPS

    cv2.imshow("Img", img)  # Show the image in a window

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
