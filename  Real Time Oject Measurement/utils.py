import cv2
import numpy as np
from numpy.ma.testutils import approx


def getContours(img,cannyThreshold=[100,100],showCanny=False,minArea=1000,filter=0,draw=False):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,cannyThreshold[0],cannyThreshold[1])
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=3)
    imgThreshold = cv2.erode(imgDial,kernel,iterations=2)
    if showCanny: cv2.imshow('Canny',imgThreshold)

    contours, hiearchy = cv2.findContours(imgThreshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    finalCountorurs = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            parameter = cv2.arcLength(i,True)
            approximate = cv2.approxPolyDP(i,0.02*parameter,True)
            bbox = cv2.boundingRect(approximate)
            if filter > 0:
                if len(approximate) == filter:
                    finalCountorurs.append([len(approximate),area,approximate,bbox,i])
            else:
                finalCountorurs.append([len(approximate),area,approximate,bbox,i])
    finalCountorurs = sorted(finalCountorurs,key = lambda x:x[1] , reverse=True)

    if draw:
        for con in finalCountorurs:
            cv2.drawContours(img,con[4],-1,(0,0,255),3)

    return img, finalCountorurs

def reorder(myPoints):
    print(myPoints.shape)
    myPointsNew = np.zeros_like(myPoints)
    myPoints = myPoints.reshape(4,2)
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]
    myPointsNew[2] = myPoints[np.argmax(diff)]
    return myPointsNew


def warpImg (img,points,w,h,pad=20):
    #print(points)
    points = reorder(points)
    points1 = np.float32(points)
    points2 = np.float32([[0,0] , [w,0] , [0,h] , [w,h]])
    matrix = cv2.getPerspectiveTransform(points1,points2)
    imgWarp = cv2.warpPerspective(img, matrix, (w,h))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad, pad:imgWarp.shape[1]-pad]

    return imgWarp

def findDistance(points1, points2):
    return ((points2[0]-points1[0])**2 + (points2[1]-points1[1])**2)**0.5
