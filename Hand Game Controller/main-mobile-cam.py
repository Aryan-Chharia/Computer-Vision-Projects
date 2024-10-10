import cv2
import mediapipe as mp

from pynput.keyboard import Controller

mp_hands = mp.solutions.hands.Hands()
keyboard = Controller()

url = 'http://<YOUR-IP>/video'
cp = cv2.VideoCapture(url)
x1, x2, y1, y2 =0, 0, 0, 0

while(True):

    _, image = cp.read()

    image_height, image_width, image_depth = image.shape
    image = cv2.flip(image, 1)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_hands = mp_hands.process(rgb_img)
    all_hands = output_hands.multi_hand_landmarks

    if all_hands:
        hand = all_hands[0]
        one_hand_landmark = hand.landmark

        for id, lm in enumerate(one_hand_landmark):
            x = int(lm.x * image_width)
            y = int(lm.y * image_height)

            if id == 12:
                x1 = x
                y1 = y

            if id == 0:
                x2 = x
                y2 = y

        distX = 0
        distX = x1 - x2
        distY = 0
        distY =y1 - y2

        if distY > -140 and distY !=0:
            # press S
            keyboard.release('d')
            keyboard.release('a')
            keyboard.release('w')
            keyboard.press('s')
            print("S")

        if distY < -200 and distY != 0:
            keyboard.release('s')
            keyboard.release('d')
            keyboard.release('a')
            keyboard.press('w')
            print("W")

        if (distX < -100 and distX != 0):
            keyboard.release('s')
            keyboard.release('d')
            keyboard.press('w')
            keyboard.press('a')
            print('A')

        if (distX > 55 and distX != 0):
            keyboard.release('a')
            keyboard.release('s')
            keyboard.press('w')
            keyboard.press('d')
            print('D')

    else:
        print('none')
        keyboard.release('d')
        keyboard.release('a')
        keyboard.release('w')
        keyboard.release('s')

    # if image is not None:
    #     cv2.imshow("Frame", image)
    q = cv2.waitKey(1)
    if q==ord("q"):
        break
cv2.destroyAllWindows()