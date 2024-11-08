import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep, time
import numpy as np
import cvzone
from pynput.keyboard import Controller
import os
from datetime import datetime

# Create a folder for saving files if it doesn't exist
if not os.path.exists("typed_texts"):
    os.makedirs("typed_texts")

# initializing the video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width of the video capture 
cap.set(4, 720)   # height of the video capture

# Increased detection confidence for better sensitivity
detector = HandDetector(detectionCon=0.8, maxHands=2)

keys = [["Q","W","E","R","T","Y","U","I","O","P"],
        ["A","S","D","F","G","H","J","K","L",";"],
        ["Z","X","C","V","B","N","M",",",".","/"],
        ["SAVE", " ", "CLEAR"]]
finalText = ""

keyboard = Controller()

class Button():
    def __init__(self, pos, text, size=[60, 60]):
        self.pos = pos
        self.size = size
        self.text = text

buttonList = []

# Update button positions with new special buttons
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        if key == " ":
            buttonList.append(Button([80 * j + 200, 80 * i + 200], key, [300, 60]))
        elif key == "SAVE":
            buttonList.append(Button([80 * j + 100, 80 * i + 200], key, [150, 60]))
        elif key == "CLEAR":
            buttonList.append(Button([80 * j + 450, 80 * i + 200], key, [160, 60]))
        else:
            buttonList.append(Button([80 * j + 100, 80 * i + 200], key))

def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (x, y, w, h), 20, rt=0)
        
        # Different colors for special buttons
        if button.text == "SAVE":
            color = (0, 255, 0)  # Green for save
        elif button.text == "CLEAR":
            color = (0, 0, 255)  # Red for clear
        else:
            color = (0, 0, 0)    # Black for regular keys
            
        cv2.rectangle(img, (x, y), (x + w, y + h), color, cv2.FILLED)
        cv2.putText(img, button.text, (x + 15, y + 45), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
    return img

def save_text_to_file(text):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"typed_texts/typed_text_{timestamp}.txt"
    with open(filename, "w") as f:
        f.write(text)
    return filename

def check_fist(hand_landmarks):
    # Check if all fingers are closed (fist gesture)
    fingers = detector.fingersUp(hand_landmarks)
    return sum(fingers) == 0  # Returns True if all fingers are down (fist)

def check_pinch(lmList):
    # Check for pinch between any two fingers with increased sensitivity
    pinch_detected = False
    # Check combinations of finger tips (landmarks 4,8,12,16,20)
    finger_tips = [4, 8, 12, 16, 20]
    
    for i in range(len(finger_tips)):
        for j in range(i + 1, len(finger_tips)):
            dist = detector.findDistance(
                (lmList[finger_tips[i]][0], lmList[finger_tips[i]][1]),
                (lmList[finger_tips[j]][0], lmList[finger_tips[j]][1])
            )[0]
            if dist < 40:  # Increased sensitivity (was 30)
                pinch_detected = True
                break
    return pinch_detected

# Initialize last click time for debouncing
last_click_time = time()
CLICK_DELAY = 0.3  # Delay between clicks

while True:
    try:
        success, img = cap.read()
        if not success:
            print("Failed to capture frame")
            continue
            
        img = cv2.flip(img, 1)
        hands, img = detector.findHands(img)
        img = drawAll(img, buttonList)

        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]

            # Check for fist gesture
            if check_fist(hand1):
                # Save current text before closing
                if finalText:
                    filename = save_text_to_file(finalText)
                    print(f"Final text saved to {filename}")
                # Display closing message
                cv2.putText(img, "Closing Program...", (400, 50), 
                           cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 255), 3)
                cv2.imshow("Image", img)
                cv2.waitKey(1000)  # Show message for 1 second
                break

            for button in buttonList:
                x, y = button.pos
                w, h = button.size

                if x < lmList1[8][0] < x+w and y < lmList1[8][1] < y+h:
                    cv2.rectangle(img, button.pos, (x+w, y+h), (175,0,175), cv2.FILLED)
                    cv2.putText(img, button.text, (x+15,y+45), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 4)

                    # Check for pinch with increased sensitivity
                    current_time = time()
                    if check_pinch(lmList1) and (current_time - last_click_time) > CLICK_DELAY:
                        if button.text == "SAVE":
                            if finalText:
                                filename = save_text_to_file(finalText)
                                print(f"Saved to {filename}")
                                cv2.putText(img, "Saved!", (500, 50), 
                                          cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                        elif button.text == "CLEAR":
                            finalText = ""
                        else:
                            keyboard.press(button.text)
                            finalText += button.text
                            
                        last_click_time = current_time
                        sleep(0.3)  # Consistent delay after each action

        # Display text input area
        cv2.rectangle(img, (150, 50), (1000, 150), (0,0,0), cv2.FILLED)
        cv2.putText(img, finalText, (160, 130), 
                    cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

        # Display instructions
        cv2.putText(img, "Make a fist to close", (50, 680), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        cv2.imshow("Image", img)
        if cv2.waitKey(5) & 0xFF == 27:
            if finalText:
                filename = save_text_to_file(finalText)
                print(f"Final text saved to {filename}")
            break

    except Exception as e:
        print(f"Error occurred: {e}")
        continue

cap.release()
cv2.destroyAllWindows()