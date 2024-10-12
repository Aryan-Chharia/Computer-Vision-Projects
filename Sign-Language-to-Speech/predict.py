from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import os
from tensorflow.keras.models import model_from_json
from matplotlib.pyplot import imshow
import matplotlib.pyplot as plt
from os import listdir
from gtts import gTTS
from playsound import playsound

def function(img):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),1)
  th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
  ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  return res

d = {0: '0', 1: 'A', 2: 'B', 3: 'C',
    4: 'D', 5: 'E', 6: 'F', 7: 'G',
    8: 'H', 9: 'I', 10: 'J', 11: 'K',
    12: 'L', 13: 'M', 14: 'N', 15: 'O',
    16: 'P', 17: 'Q', 18: 'R', 19: 'S',
    20: 'T', 21: 'U', 22: 'V', 23: 'W',
    24: 'X', 25: 'Y',26: 'Z'}

json_file = open('model-bw.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model-bw.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

upper_left = (335, 55)
bottom_right = (635, 355)

cam = cv2.VideoCapture(0)
cv2.namedWindow("test", cv2.WINDOW_NORMAL)
cv2.resizeWindow('test', 1050,1250)

a=""
g=""
i=1
while True:
    ret, frame = cam.read()
    frame= cv2.flip(frame, 1)
    r = cv2.rectangle(frame, upper_left, bottom_right, (0, 0, 0), 5)
    rect_img = frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]]
    sketcher_rect = rect_img
    sketcher_rect = function(sketcher_rect)
    sketcher_rect_rgb = cv2.cvtColor(sketcher_rect, cv2.COLOR_GRAY2RGB)
    frame[upper_left[1] : bottom_right[1], upper_left[0] : bottom_right[0]] = sketcher_rect_rgb
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print(g)
        tts = gTTS(g,lang='en') #Provide the string to convert to speech
        m=str(i)+".mp3"
        tts.save(m) #save the string converted to speech as a .wav file
        playsound(m)
        i=i+1
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "frame_0.png"
        sketcher_rect = cv2.resize(sketcher_rect,(128, 128))
        print(sketcher_rect.shape)
        cv2.imwrite(img_name, sketcher_rect)
        print("{} written!".format(img_name))
        x = image.img_to_array(sketcher_rect)
        x = np.expand_dims(x, axis=0)
        x = x/255.0
        pre = loaded_model.predict(x)
        p_test=np.argmax(pre)
        a = d[p_test]
        g=g+a
        print(a)
        tts = gTTS(a,lang='en') #Provide the string to convert to speech
        m=str(i)+".mp3"
        tts.save(m) #save the string converted to speech as a .wav file
        playsound(m)
        i=i+1
        # Audio(sound_file, autoplay=True)
cam.release()

cv2.destroyAllWindows()

