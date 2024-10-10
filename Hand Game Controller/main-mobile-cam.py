import cv2
import mediapipe as mp
from pynput.keyboard import Controller

# Initialize MediaPipe hands module
mp_hands = mp.solutions.hands.Hands()  # Default settings for hand detection
keyboard = Controller()  # Initialize keyboard controller to simulate key presses

# URL of the video stream (replace <YOUR-IP> with the actual IP)
url = 'http://<YOUR-IP>/video'
cp = cv2.VideoCapture(url)  # Capture video from the network stream URL

# Initialize coordinates for hand landmarks
x1, x2, y1, y2 = 0, 0, 0, 0

while(True):
    # Read frame from the video stream
    _, image = cp.read()
    
    # Get the dimensions of the image
    image_height, image_width, image_depth = image.shape
    
    # Flip the image horizontally to create a mirror effect (selfie view)
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB (required for MediaPipe processing)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image to detect hand landmarks
    output_hands = mp_hands.process(rgb_img)
    all_hands = output_hands.multi_hand_landmarks  # Get detected hand landmarks
    
    if all_hands:
        hand = all_hands[0]  # Take the first detected hand
        one_hand_landmark = hand.landmark  # Get the list of hand landmarks

        # Iterate over hand landmarks and track specific points
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

        # Map hand movements to key presses:
        if distY > -140 and distY != 0:  # Hand moving down (press 'S')
            keyboard.release('d')
            keyboard.release('a')
            keyboard.release('w')
            keyboard.press('s')
            print("S")

        if distY < -200 and distY != 0:  # Hand moving up (press 'W')
            keyboard.release('s')
            keyboard.release('d')
            keyboard.release('a')
            keyboard.press('w')
            print("W")

        if distX < -100 and distX != 0:  # Hand moving left (press 'A')
            keyboard.release('s')
            keyboard.release('d')
            keyboard.press('w')
            keyboard.press('a')
            print('A')

        if distX > 55 and distX != 0:  # Hand moving right (press 'D')
            keyboard.release('a')
            keyboard.release('s')
            keyboard.press('w')
            keyboard.press('d')
            print('D')

    else:
        # If no hands are detected, release all keys
        print('none')
        keyboard.release('d')
        keyboard.release('a')
        keyboard.release('w')
        keyboard.release('s')

    # Optionally display the frame (commented out for now)
    # if image is not None:
    #     cv2.imshow("Frame", image)

    # Check if 'q' is pressed to exit
    q = cv2.waitKey(1)
    if q == ord("q"):
        break

# Close all OpenCV windows and release the video stream
cv2.destroyAllWindows()
