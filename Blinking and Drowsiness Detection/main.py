import numpy as np
import cv2
import dlib
from math import sqrt
from imutils import face_utils


MIN_EAR = 0.2
MIN_DROWSY_EAR = 0.3
MAX_DROWSY_FRAMES = 35
LEFT_EYE_START = 36
LEFT_EYE_END = 42
RIGHT_EYE_START = 42
RIGHT_EYE_END = 48


def midpoint(pt1, pt2):
    return int((pt1.x + pt2.x) / 2), int((pt2.y + pt1.y) / 2)


def distance(pt1, pt2):
    return sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def eye_aspect_ratio(eye_landmarks, facial_landmarks):
    left_corner = (
        facial_landmarks.part(eye_landmarks[0]).x,
        facial_landmarks.part(eye_landmarks[0]).y,
    )
    right_corner = (
        facial_landmarks.part(eye_landmarks[3]).x,
        facial_landmarks.part(eye_landmarks[3]).y,
    )
    top_left = (
        facial_landmarks.part(eye_landmarks[1]).x,
        facial_landmarks.part(eye_landmarks[1]).y,
    )
    top_right = (
        facial_landmarks.part(eye_landmarks[2]).x,
        facial_landmarks.part(eye_landmarks[2]).y,
    )
    bottom_right = (
        facial_landmarks.part(eye_landmarks[4]).x,
        facial_landmarks.part(eye_landmarks[4]).y,
    )
    bottom_left = (
        facial_landmarks.part(eye_landmarks[5]).x,
        facial_landmarks.part(eye_landmarks[5]).y,
    )
    return (distance(top_left, bottom_left) + distance(top_right, bottom_right)) / (
        2 * distance(left_corner, right_corner)
    )


cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./Dataset/shape_predictor_68_face_landmarks.dat")
left_eye = [x for x in range(36, 42)]
right_eye = [x for x in range(42, 48)]
drowsy_count = 0

while True:
    ret, frame = cap.read()
    faces, a, b = detector.run(image=frame, upsample_num_times=0, adjust_threshold=0.0)
    
    for face in faces:
        landmarks = predictor(frame, face)
        left_eye_ratio = eye_aspect_ratio(left_eye, landmarks)
        right_eye_ratio = eye_aspect_ratio(right_eye, landmarks)
        blink_ratio = (left_eye_ratio + right_eye_ratio) / 2
        landmarks = face_utils.shape_to_np(landmarks)
        left_eye_hull = cv2.convexHull(landmarks[LEFT_EYE_START:LEFT_EYE_END])
        right_eye_hull = cv2.convexHull(landmarks[RIGHT_EYE_START:RIGHT_EYE_END])
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), 2)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 2)
        if cv2.waitKey(1) == ord("q"):
            exit
        if blink_ratio < MIN_EAR:
            cv2.putText(
                frame,
                "You are blinking.",
                (50, 100),
                cv2.FONT_HERSHEY_DUPLEX,
                2,
                (0, 0, 255),
                3,
                cv2.LINE_8,
            )
        if blink_ratio < MIN_DROWSY_EAR:
            drowsy_count += 1
        if drowsy_count > MAX_DROWSY_FRAMES:
            cv2.putText(
                frame,
                "You are drowsy.",
                (50, 400),
                cv2.FONT_HERSHEY_DUPLEX,
                2,
                (255, 0, 0),
                3,
                cv2.LINE_8,
            )
            drowsy_count = 0
    cv2.imshow("You", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
