# install requeried libraries
#!pip install opencv-python 
#!pip install mediapipe 
#!pip install screen-brightness-control 
#!pip install pycaw 
#!pip install comtypes

import cv2
import numpy as np
import mediapipe as mp
from screen_brightness_control import set_brightness
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize Mediapipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Setup audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # Get thumb and index tip positions
            thumb_tip = hand_lms.landmark[4]
            index_tip = hand_lms.landmark[8]

            h, w, _ = img.shape
            thumb_coords = (int(thumb_tip.x * w), int(thumb_tip.y * h))
            index_coords = (int(index_tip.x * w), int(index_tip.y * h))

            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_lms, mp_hands.HAND_CONNECTIONS)

            # Calculate horizontal (x-axis) and vertical (y-axis) distances
            horizontal_distance = np.abs(thumb_coords[0] - index_coords[0])  # X distance for volume
            vertical_distance = np.abs(thumb_coords[1] - index_coords[1])  # Y distance for brightness

            # Normalize the horizontal distance to the volume range
            vol = np.interp(horizontal_distance, [30, 300], [min_vol, max_vol])
            volume.SetMasterVolumeLevel(vol, None)

            # Normalize the vertical distance to brightness control
            brightness = np.interp(vertical_distance, [30, 300], [0, 100])
            set_brightness(int(brightness))

            # Display controls
            cv2.putText(img, f'Volume: {int((vol - min_vol) / (max_vol - min_vol) * 100)}%', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(img, f'Brightness: {int(brightness)}%', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Img', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()