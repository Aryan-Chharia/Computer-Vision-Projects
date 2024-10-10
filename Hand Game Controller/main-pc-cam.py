import cv2
import mediapipe as mp
from pynput.keyboard import Controller

# Initialize MediaPipe hands with optimizations
mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
keyboard = Controller()

# Open the camera
cp = cv2.VideoCapture(0)
x1, x2, y1, y2 = 0, 0, 0, 0
pressed_key = ""
frame_skip = 1  # Reduce frame skipping to 1 for smoother processing

frame_count = 0

while True:
    # Capture frame from camera
    ret, image = cp.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    frame_count += 1

    # Skip every nth frame to improve performance (reduced skipping)
    if frame_count % frame_skip != 0:
        continue

    # Get image dimensions (without resizing)
    image_height, image_width, _ = image.shape

    # Flip the image horizontally for a selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the image to RGB
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image to detect hands
    output_hands = mp_hands.process(rgb_img)
    all_hands = output_hands.multi_hand_landmarks

    # Detect keypresses based on hand position
    if all_hands:
        hand = all_hands[0]
        one_hand_landmark = hand.landmark

        for id, lm in enumerate(one_hand_landmark):
            x = int(lm.x * image_width)
            y = int(lm.y * image_height)

            if id == 12:  # Finger tip of middle finger
                x1 = x
                y1 = y

            if id == 0:  # Wrist point
                x2 = x
                y2 = y

        distX = x1 - x2
        distY = y1 - y2

        if distY > -140 and distY != 0:
            keyboard.release('d')
            keyboard.release('a')
            keyboard.release('w')
            keyboard.press('s')
            print("Pressed Key: S")
        elif distY < -200 and distY != 0:
            keyboard.release('s')
            keyboard.release('d')
            keyboard.release('a')
            keyboard.press('w')
            print("Pressed Key: W")
        elif distX < -100 and distX != 0:
            keyboard.release('s')
            keyboard.release('d')
            keyboard.press('w')
            keyboard.press('a')
            print("Pressed Key: A")
        elif distX > 55 and distX != 0:
            keyboard.release('a')
            keyboard.release('s')
            keyboard.press('w')
            keyboard.press('d')
            print("Pressed Key: D")
    else:
        keyboard.release('d')
        keyboard.release('a')
        keyboard.release('w')
        keyboard.release('s')

    # Display the camera feed
    cv2.imshow("Camera Feed", image)

    # Check if 'q' is pressed to quit
    q = cv2.waitKey(1)
    if q == ord("q"):
        break

# Release resources
cv2.destroyAllWindows()
cp.release()
