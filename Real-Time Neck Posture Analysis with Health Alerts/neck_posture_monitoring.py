import cv2
import dlib
import numpy as np
from imutils import face_utils

# Load pre-trained models for facial landmark detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Function to calculate angle between two points (neck and vertical axis)
def calculate_angle(point1, point2):
    delta_y = point2[1] - point1[1]
    delta_x = point2[0] - point1[0]
    angle = np.arctan2(delta_y, delta_x) * 180.0 / np.pi
    return angle

# Function to monitor the neck angle and posture
def monitor_posture(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        # Get keypoints for neck and shoulders
        chin = landmarks[8]     # Chin point
        left_shoulder = landmarks[1]  # Leftmost point near jaw
        right_shoulder = landmarks[15] # Rightmost point near jaw

        # Calculate the midpoint between left and right shoulders
        midpoint = [(left_shoulder[0] + right_shoulder[0]) // 2, 
                    (left_shoulder[1] + right_shoulder[1]) // 2]

        # Calculate the neck angle with the vertical axis
        neck_angle = calculate_angle(chin, midpoint)

        # Draw the landmarks on the frame
        cv2.circle(frame, tuple(chin), 3, (0, 255, 0), -1)
        cv2.circle(frame, tuple(midpoint), 3, (255, 0, 0), -1)
        cv2.line(frame, tuple(chin), tuple(midpoint), (255, 0, 0), 2)

        # Check for poor posture (angle greater than 15 degrees)
        if abs(neck_angle) > 15:
            cv2.putText(frame, "Poor Posture Detected!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Good Posture", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

# Capture video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = monitor_posture(frame)

    # Display the resulting frame
    cv2.imshow('Posture Monitor', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
