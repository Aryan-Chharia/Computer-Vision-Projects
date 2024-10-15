import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import string

def func(path):
  img = cv2.imread(path)
  img=cv2.resize(img,(300, 300))
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(gray,(5,5),2)
  th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
  ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
  return res

if not os.path.exists("Sign2"):
    os.makedirs("Sign2")
if not os.path.exists("Sign2/train"):
    os.makedirs("Sign2/train")
if not os.path.exists("Sign2/test"):
    os.makedirs("Sign2/test")

l=[]
path="Sign/train"
path2="Sign2/train"
for (dirpath,dirnames,filenames) in os.walk(path):
  dirnames.sort()
  for dirname in dirnames:
    for (direcpath,direcnames,files) in os.walk(path+"/"+dirname):
      if not os.path.exists(path2+"/"+dirname):
        os.makedirs(path2+"/"+dirname)
      for file in files:
        if dirname not in l:
          l.append(dirname)
        actual_path=path+"/"+dirname+"/"+file #path of each image
        actual_path2=path2+"/"+dirname+"/"+file
        bw_img=func(actual_path) 
        cv2.imwrite(actual_path2,bw_img)

path="Sign/test"
path2="Sign2/test"
for (dirpath,dirnames,filenames) in os.walk(path):
  dirnames.sort()
  for dirname in dirnames:
    for (direcpath,direcnames,files) in os.walk(path+"/"+dirname):
      if not os.path.exists(path2+"/"+dirname):
        os.makedirs(path2+"/"+dirname)
      for file in files:
        if dirname not in l:
          l.append(dirname)
        actual_path=path+"/"+dirname+"/"+file #path of each image
        actual_path2=path2+"/"+dirname+"/"+file
        bw_img=func(actual_path) 
        cv2.imwrite(actual_path2,bw_img)

print("Done")