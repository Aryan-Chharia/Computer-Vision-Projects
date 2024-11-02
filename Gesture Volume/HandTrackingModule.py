import cv2
import mediapipe as mp
import time
import math
import pyautogui
import tkinter as tk

class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=1, detection_confidence=0.5, tracking_confidence=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )

        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]
        self.results = None
        self.lm_list = []

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return img

    def find_position(self, img, hand_no=0, draw=True):
        x_list, y_list = [], []
        bbox = []
        self.lm_list = []

        if self.results.multi_hand_landmarks:
            if len(self.results.multi_hand_landmarks) > hand_no:
                my_hand = self.results.multi_hand_landmarks[hand_no]

                for id, lm in enumerate(my_hand.landmark):
                    h, w, _ = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    x_list.append(cx)
                    y_list.append(cy)
                    self.lm_list.append([id, cx, cy])

                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                if x_list and y_list:
                    xmin, xmax = min(x_list), max(x_list)
                    ymin, ymax = min(y_list), max(y_list)
                    bbox = (xmin, ymin, xmax, ymax)

                    if draw:
                        cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                      (bbox[2] + 20, bbox[3] + 20), (0, 255, 0), 2)

        return self.lm_list, bbox

    def fingers_up(self):
        fingers = []
        if len(self.lm_list) >= 21:
            fingers.append(1 if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1] else 0)
            for id in range(1, 5):
                fingers.append(1 if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2] else 0)
        return fingers

    def find_distance(self, p1, p2, img, draw=True):
        if len(self.lm_list) >= max(p1, p2):
            x1, y1 = self.lm_list[p1][1], self.lm_list[p1][2]
            x2, y2 = self.lm_list[p2][1], self.lm_list[p2][2]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if draw:
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

            length = math.hypot(x2 - x1, y2 - y1)
            return length, img, [x1, y1, x2, y2, cx, cy]
        return None

class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hand Control with PyAutoGUI")
        self.geometry("400x200")
        self.label = tk.Label(self, text="Move your hand to control the cursor!", font=("Helvetica", 14))
        self.label.pack(pady=20)
        self.start_button = tk.Button(self, text="Start Tracking", command=self.start_tracking)
        self.start_button.pack(pady=10)

        self.detector = HandDetector()
        self.cap = cv2.VideoCapture(0)
        self.running = False

    def start_tracking(self):
        self.running = True
        self.track_hand()

    def track_hand(self):
        if not self.running:
            return

        success, img = self.cap.read()
        if success:
            img = self.detector.find_hands(img)
            lm_list, _ = self.detector.find_position(img)

            if lm_list:
                # Move the mouse cursor based on the position of the index finger (tip ID 8)
                index_finger_x = lm_list[8][1]
                index_finger_y = lm_list[8][2]

                # Get screen size
                screen_width, screen_height = pyautogui.size()
                # Normalize the hand position to screen dimensions
                x = int(screen_width * index_finger_x / img.shape[1])
                y = int(screen_height * index_finger_y / img.shape[0])

                # Move the cursor
                pyautogui.moveTo(x, y)

            # Display the image in a window (optional)
            cv2.imshow("Hand Tracking", img)

        self.after(10, self.track_hand)

    def on_closing(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()
        self.destroy()

if __name__ == "__main__":
    app = Application()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
