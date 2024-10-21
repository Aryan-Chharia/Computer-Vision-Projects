import cv2
import mediapipe as mp
import pyautogui

# Initialize camera and face mesh model
cam = cv2.VideoCapture(0)
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Get screen dimensions
screen_w, screen_h = pyautogui.size()

while True:
    # Capture frame
    _, frame = cam.read()
    frame = cv2.flip(frame, 1)  # Flip the frame horizontally
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB

    # Process the frame to find landmarks
    output = face_mesh.process(rgb_frame)
    landmark_points = output.multi_face_landmarks
    frame_h, frame_w, _ = frame.shape

    # Check if landmarks are detected
    if landmark_points:
        landmarks = landmark_points[0].landmark

        # Draw eye landmarks
        for id, landmark in enumerate(landmarks[474:478]):
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 0))  # Draw landmark

            # Map landmark coordinates to screen coordinates
            if id == 1:  # Typically the right eye landmark
                # Normalize the coordinates to the screen size
                screen_x = int(screen_w * landmark.x)
                screen_y = int(screen_h * landmark.y)

                # Move mouse to the new position
                pyautogui.moveTo(screen_x, screen_y)

        # Detect blinking (click action)
        left_eye = [landmarks[145], landmarks[159]]  # Landmarks for left eye
        for landmark in left_eye:
            x = int(landmark.x * frame_w)
            y = int(landmark.y * frame_h)
            cv2.circle(frame, (x, y), 3, (0, 255, 255))

        # Click if the eyes are closed (or some other condition you define)
        if (left_eye[0].y - left_eye[1].y) < 0.004:  # Adjust this threshold as necessary
            pyautogui.click()
            pyautogui.sleep(1)  # Sleep to avoid multiple clicks

    # Show the frame with landmarks
    cv2.imshow('Eye Controlled Mouse', frame)
    
    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
