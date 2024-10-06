import cv2
import os
from cvzone.HandTrackingModule import HandDetector

import tkinter as tk

width, height = 800, 600  # Adjust the width and height to a smaller size
folderPath = "Images"

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

pathImages = sorted(os.listdir(folderPath))
print(pathImages)

imgNumber = 0
hs, ws = int(height * 0.15), int(width * 0.25)

answer = [[0, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 1, 0, 0, 0], [0, 1, 1, 1, 0], [0, 1, 1, 0, 0]]
detector = HandDetector(detectionCon=0.8, maxHands=1)
buttonPressed = False
buttonCounter = 0
buttonDelay = 100
submit=[]
def display_score(score):
    root = tk.Tk()
    root.geometry("300x200")
    
    score_label = tk.Label(root, text=f"Your Score: {score}", font=("Arial", 24))
    score_label.pack(pady=50)
    
    root.mainloop()
def check_result(submit,answer):
    count=0
    for a in range(0,len(answer)):
        if submit[a]==answer[a]:
            count+=1
    return count
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgCurrent = cv2.imread(os.path.join(folderPath, pathImages[imgNumber]))
    imgCurrent = cv2.resize(imgCurrent, (width, height))

    hands, _ = detector.findHands(img)
    if hands and not buttonPressed:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmlist = hand['lmList']

        if fingers == [0, 1, 0, 0, 0]:
            submit.append(fingers)
            if imgNumber < len(pathImages) - 1:
                imgNumber += 1
                buttonPressed = True
            else:
                r=check_result(submit,answer)
                display_score(r)
                break
        if fingers == [0, 1, 1, 0, 0]:
            submit.append(fingers)
            if imgNumber < len(pathImages) - 1:
                imgNumber += 1
                buttonPressed = True
            else:
                r=check_result(submit,answer)
                display_score(r)
                break
        if fingers == [0, 1, 1, 1, 0]:
            submit.append(fingers)
            if imgNumber < len(pathImages) - 1:
                imgNumber += 1
                buttonPressed = True
            else:
                r=check_result(submit,answer)
                display_score(r)
                break
        if fingers == [0, 1, 1, 1, 1]:
            submit.append(fingers)
            if imgNumber < len(pathImages) - 1:
                imgNumber += 1
                buttonPressed = True
            else:
                r=check_result(submit,answer)
                display_score(r)
                break
        if fingers == [0, 0, 0, 0, 0]:
            pass

    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape

    imgCurrent[0:hs, w - ws : w] = imgSmall
    cv2.imshow("Slides", imgCurrent)
    key = cv2.waitKey(1)
    if key == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
