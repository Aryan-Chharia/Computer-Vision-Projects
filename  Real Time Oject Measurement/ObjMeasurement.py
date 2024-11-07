import cv2
import numpy as np
import utils
from utils import warpImg

webcam = False
path = '1.jpeg'
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)
scaleFactor = 3

wPaper = 210*scaleFactor
hPaper = 297*scaleFactor

while True:
    if webcam:success,img = cap.read()
    else: img = cv2.imread(path)

    imgContours, finalCountours = utils.getContours(img,minArea=50000,filter=4)

    if len(finalCountours) != 0:
        biggest = finalCountours[0][2]
        #print(biggest)
        imgWarp = utils.warpImg(img, biggest,wPaper, hPaper )
        imgContours2, finalCountours2 = utils.getContours(imgWarp, minArea=2000, filter=4,cannyThreshold=[50,50],draw=False)
        if len(finalCountours) != 0:
            for obj in finalCountours2:
                cv2.polylines(imgContours2,[obj[2]],True,(0,255,0),2)
                newPoints = utils.reorder(obj[2])
                newWidth = round(utils.findDistance(newPoints[0][0]//scaleFactor,newPoints[1][0]//scaleFactor)/10),1
                newHeight = round(utils.findDistance(newPoints[0][0]//scaleFactor,newPoints[2][0]//scaleFactor)/10),1
                cv2.arrowedLine(imgContours2 , (newPoints[0][0][0],newPoints[0][0][1]) , (newPoints[1][0][0],newPoints[1][0][1]) , (255,0,255), 3 , 8 , 0 , 0.05)
                cv2.arrowedLine(imgContours2 , (newPoints[0][0][0],newPoints[0][0][1]) , (newPoints[2][0][0],newPoints[2][0][1]) , (255,0,255), 3 , 8 , 0 , 0.05)
                x, y, w , h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(newWidth) , (x + 30, y - 10) , cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255,0,255),2)
                cv2.putText(imgContours2, '{}cm'.format(newHeight) , (x - 70, y + h // 2) , cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255,0,255),2)


        cv2.imshow('A4 Paper', imgContours2)


    img = cv2.resize(img,(0,0),None,0.5,0.5)
    cv2.imshow('Original',img)
    cv2.waitKey(1)

