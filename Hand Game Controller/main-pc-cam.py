import cv2
import mediapipe as mp
from pynput.keyboard import Controller

# Initialize MediaPipe hands for hand tracking with optimizations for real-time use
mp_hands = mp.solutions.hands.Hands(static_image_mode=False,  # False for real-time video
                                    max_num_hands=1,           # Detect only one hand
                                    min_detection_confidence=0.5,  # Minimum confidence for detection
                                    min_tracking_confidence=0.5)  # Minimum confidence for hand tracking
keyboard = Controller()  # Initialize keyboard controller to simulate key presses

# Open the camera
cp = cv2.VideoCapture(0)  # Capture video from the default camera (index 0)
x1, x2, y1, y2 = 0, 0, 0, 0  # Initialize coordinates for hand landmarks
pressed_key = ""  # Variable to track the currently pressed key
frame_skip = 1  # Process every frame (set to 1 for smoother processing)

frame_count = 0  # Counter to track frames

while True:
    # Capture frame from camera
    ret, image = cp.read()  # Read frame from the camera
    if not ret:
        print("Failed to capture frame. Exiting...")  # Exit if no frame is captured
        break

    frame_count += 1  # Increment frame counter

    # Skip every nth frame to reduce computation (frame skipping)
    if frame_count % frame_skip != 0:
        continue  # Continue to the next loop without processing the current frame

    # Get image dimensions (used for scaling landmarks)
    image_height, image_width, _ = image.shape

    # Flip the image horizontally to create a selfie-view
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB (required for MediaPipe processing)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image to detect hands
    output_hands = mp_hands.process(rgb_img)
    all_hands = output_hands.multi_hand_landmarks  # Get hand landmarks

    # Detect keypresses based on hand positions (if hands are detected)
    if all_hands:
        hand = all_hands[0]  # Take the first detected hand
        one_hand_landmark = hand.landmark  # Get the list of hand landmarks

        # Iterate over hand landmarks and track positions of specific points
        for id, lm in enumerate(one_hand_landmark):
            # Scale the landmark coordinates to the image size
            x = int(lm.x * image_width)
            y = int(lm.y * image_height)

            # Track the middle finger tip (id=12)
            if id == 12:  
                x1 = x
                y1 = y

            # Track the wrist point (id=0)
            if id == 0:
                x2 = x
                y2 = y

        # Calculate the distance between the wrist and middle finger tip
        distX = x1 - x2
        distY = y1 - y2

        # Map hand movement to key presses:
        if distY > -140 and distY != 0:  # Hand moving down (press 'S')
            keyboard.release('d')
            keyboard.release('a')
            keyboard.release('w')
            keyboard.press('s')
            print("Pressed Key: S")
        elif distY < -200 and distY != 0:  # Hand moving up (press 'W')
            keyboard.release('s')
            keyboard.release('d')
            keyboard.release('a')
            keyboard.press('w')
            print("Pressed Key: W")
        elif distX < -100 and distX != 0:  # Hand moving left (press 'A')
            keyboard.release('s')
            keyboard.release('d')
            keyboard.press('w')
            keyboard.press('a')
            print("Pressed Key: A")
        elif distX > 55 and distX != 0:  # Hand moving right (press 'D')
            keyboard.release('a')
            keyboard.release('s')
            keyboard.press('w')
            keyboard.press('d')
            print("Pressed Key: D")
    else:
        # Release all keys if no hands are detected
        keyboard.release('d')
        keyboard.release('a')
        keyboard.release('w')
        keyboard.release('s')

    # Display the camera feed in a window
    cv2.imshow("Camera Feed", image)

    # Break the loop if 'q' key is pressed
    q = cv2.waitKey(1)
    if q == ord("q"):
        break

# Release resources
cv2.destroyAllWindows()  # Close all OpenCV windows
cp.release()  # Release the camera resource
