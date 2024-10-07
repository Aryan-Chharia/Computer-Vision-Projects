import cv2
import numpy as np
from pyzbar.pyzbar import decode # Decodes barcodes/QR codes.

cap = cv2.VideoCapture(0) # Opens the default webcam.
cap.set(3,640)
cap.set(4,480)

while True:

    success, img = cap.read()
    for barcode in decode(img):
        myData = barcode.data.decode('utf-8')
        print(myData)
        with open("result.txt", "a") as file:
            file.write(myData + "\n")                   #Write scanned data into text file
        pts = np.array([barcode.polygon],np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(255,0,255),5) # Draws a polygon around the detected barcode/QR code.
        pts2 = barcode.rect
        cv2.putText(img,myData,(pts2[0],pts2[1]),cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255,0,255),2) # Displays the decoded data on the image.

    cv2.imshow('Result',img)
    cv2.waitKey(1)