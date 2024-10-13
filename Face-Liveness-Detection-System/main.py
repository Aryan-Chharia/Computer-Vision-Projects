import cv2
import dlib # type: ignore
import numpy as np
from imutils import face_utils # type: ignore
from scipy.spatial import distance as dist
import time
import os
from playsound import playsound  # type: ignore # You may need to install this library

# Parameters for Blink Detection
EYE_AR_THRESH = 0.23  # Eye Aspect Ratio threshold
EYE_AR_CONSEC_FRAMES = 1  # Consecutive frames below threshold

# Load Haar Cascade Classifiers
frontal_face_cascade = cv2.CascadeClassifier('dataset/haarcascade_frontalface_default.xml')
profile_face_cascade = cv2.CascadeClassifier('dataset/haarcascade_profileface.xml')

# Load Dlib shape predictor for eye landmarks
predictor_path = "dataset/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def detect_blink(frame, rect):
    shape = predictor(frame, rect)
    shape_np = face_utils.shape_to_np(shape)  # Convert shape to NumPy array

    left_eye = shape_np[36:42]  # Left eye landmarks
    right_eye = shape_np[42:48]  # Right eye landmarks

    left_EAR = eye_aspect_ratio(left_eye)
    right_EAR = eye_aspect_ratio(right_eye)

    ear = (left_EAR + right_EAR) / 2.0
    return ear

def save_capture(frame, blink_count):
    """Save the captured frame with the blink count as the filename."""
    if not os.path.exists('captures'):
        os.makedirs('captures')
    filename = f'captures/blink_capture_{blink_count}.jpg'
    cv2.imwrite(filename, frame)
    print(f"Saved: {filename}")

def play_blink_sound():
    """Play a sound when a blink is detected."""
    # Specify the path to your sound file here
    playsound('path_to_your_sound_file.mp3')  # Ensure you have a sound file in the specified path

def main():
    cap = cv2.VideoCapture(0)
    COUNTER = 0
    TOTAL = 0
    start_time = time.time()  # Start time for calculating blink rate

    # Instructions for the user
    instructions = [
        "Welcome to the Blink Detection System!",
        "Please ensure your face is visible to the camera.",
        "You will be instructed to blink during the session.",
        "Turn your head left and blink.",
        "Turn your head right and blink.",
        "Press 'q' at any time to exit."
    ]

    for instruction in instructions:
        print(instruction)  # Print instruction for debugging
        time.sleep(2)  # Wait for 2 seconds before capturing

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Profile detection
        faces = frontal_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        profiles = profile_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        # Blink detection
        for (x, y, w, h) in faces:
            rect = dlib.rectangle(x, y, x + w, y + h)
            ear = detect_blink(gray, rect)

            # Check for blink
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                # Visual feedback for blink detection
                cv2.putText(frame, "Blink Detected!", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                play_blink_sound()  # Play sound on blink
                save_capture(frame, TOTAL + 1)  # Save capture
            else:
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    TOTAL += 1
                COUNTER = 0

            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        for (x, y, w, h) in profiles:
            # Draw rectangle around profile face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Show blink count and blink ratio on frame
        elapsed_time = time.time() - start_time
        blink_ratio = TOTAL / (elapsed_time / 60) if elapsed_time > 0 else 0
        cv2.putText(frame, f"Blinks: {TOTAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Blink Ratio: {blink_ratio:.2f} blinks/min", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Total Time: {int(elapsed_time)}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Liveness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting...")
            break

    # Check conditions for liveness
    if TOTAL > 0:
        print("Liveness Successful!")
        cv2.putText(frame, "Liveness Successful!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        print("Liveness Failed!")
        cv2.putText(frame, "Liveness Failed!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Liveness Detection", frame)
    cv2.waitKey(2000)  # Display the result for 2 seconds

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
