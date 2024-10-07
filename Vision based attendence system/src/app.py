# -*- coding: utf-8 -*-
"""
Created on Firday Jun 14 2019
@author: Jiwitesh Kumar
"""
import argparse
import logging
import tkinter as tk
from tkinter import *


import tkinter.font as font
import webbrowser
import random

from readme_renderer import txt

from src.clientApp import collectUserImageForRegistration, getFaceEmbedding, trainModel
from src.com_in_ineuron_ai_collect_trainingdata.get_faces_from_camera import TrainingDataCollector
from src.com_in_ineuron_ai_face_embedding.faces_embedding import GenerateFaceEmbedding
from src.com_in_ineuron_ai_predictor.facePredictor import FacePredictor
from src.com_in_ineuron_ai_training.train_softmax import TrainFaceRecogModel


class RegistrationModule:
    def __init__(self, logFileName):

        self.logFileName = logFileName
        self.window = tk.Tk()
        # helv36 = tk.Font(family='Helvetica', size=36, weight='bold')
        self.window.title("Face Recognition and Tracking")

        # this removes the maximize button
        self.window.resizable(0, 0)
        window_height = 600
        window_width = 880

        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()

        x_cordinate = int((screen_width / 2) - (window_width / 2))
        y_cordinate = int((screen_height / 2) - (window_height / 2))

        self.window.geometry("{}x{}+{}+{}".format(window_width, window_height, x_cordinate, y_cordinate))
        # window.geometry('880x600')
        self.window.configure(background='#ffffff')

        # window.attributes('-fullscreen', True)

        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        header = tk.Label(self.window, text="Employee Monitoring Registration", width=80, height=2, fg="white", bg="#363e75",
                          font=('times', 18, 'bold', 'underline'))
        header.place(x=0, y=0)
        clientID = tk.Label(self.window, text="Client ID", width=10, height=2, fg="white", bg="#363e75", font=('times', 15))
        clientID.place(x=80, y=80)

        displayVariable = StringVar()
        self.clientIDTxt = tk.Entry(self.window, width=20, text=displayVariable, bg="white", fg="black",
                               font=('times', 15, 'bold'))
        self.clientIDTxt.place(x=205, y=80)

        empID = tk.Label(self.window, text="EmpID", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
        empID.place(x=450, y=80)

        self.empIDTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.empIDTxt.place(x=575, y=80)

        empName = tk.Label(self.window, text="Emp Name", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
        empName.place(x=80, y=140)

        self.empNameTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.empNameTxt.place(x=205, y=140)

        emailId = tk.Label(self.window, text="Email ID :", width=10, fg="white", bg="#363e75", height=2, font=('times', 15))
        emailId.place(x=450, y=140)

        self.emailIDTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.emailIDTxt.place(x=575, y=140)

        mobileNo = tk.Label(self.window, text="Mobile No :", width=10, fg="white", bg="#363e75", height=2,
                            font=('times', 15))
        mobileNo.place(x=450, y=140)

        self.mobileNoTxt = tk.Entry(self.window, width=20, bg="white", fg="black", font=('times', 15, ' bold '))
        self.mobileNoTxt.place(x=575, y=140)

        lbl3 = tk.Label(self.window, text="Notification : ", width=15, fg="white", bg="#363e75", height=2,
                        font=('times', 15))
        self.message = tk.Label(self.window, text="", bg="white", fg="black", width=30, height=1,
                                activebackground="#e47911", font=('times', 15))
        self.message.place(x=220, y=220)
        lbl3.place(x=80, y=260)

        self.message = tk.Label(self.window, text="", bg="#bbc7d4", fg="black", width=58, height=2, activebackground="#bbc7d4",
                           font=('times', 15))
        self.message.place(x=205, y=260)

        # lbl3 = tk.Label(self.window, text="Attendance : ", width=15, fg="white", bg="#363e75", height=2,
        #                 font=('times', 15))
        # lbl3.place(x=80, y=440)
        #
        # self.message2 = tk.Label(self.window, text="", fg="#e47911", bg="#bbc7d4", activeforeground="#f8f9fa", width=52, height=2,
        #                     font=('times', 15))
        # self.message2.place(x=250, y=440)

        takeImg = tk.Button(self.window, text="Take Images", command=self.collectUserImageForRegistration, fg="white", bg="#363e75", width=15,
                            height=2,
                            activebackground="#118ce1", font=('times', 15, ' bold '))
        takeImg.place(x=80, y=350)

        trainImg = tk.Button(self.window, text="Train Images", command=self.trainModel, fg="white", bg="#363e75", width=15,
                             height=2,
                             activebackground="#118ce1", font=('times', 15, ' bold '))
        trainImg.place(x=350, y=350)

        predictImg = tk.Button(self.window, text="Predict", command=self.makePrediction, fg="white", bg="#363e75",
                             width=15,
                             height=2,
                             activebackground="#118ce1", font=('times', 15, ' bold '))
        predictImg.place(x=600, y=350)

        quitWindow = tk.Button(self.window, text="Quit", command=self.close_window, fg="white", bg="#363e75", width=10, height=2,
                               activebackground="#118ce1", font=('times', 15, 'bold'))
        quitWindow.place(x=650, y=510)

        link2 = tk.Label(self.window, text="Copyright©2020, Ineuron.ai", fg="blue", )
        link2.place(x=690, y=580)
        # link2.pack()
        link2.bind("<Button-1>", lambda e: self.callback("http://ineuron.ai"))
        label = tk.Label(self.window)

        self.window.mainloop()

        logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', filename=self.logFileName,
                            level=logging.INFO,
                            datefmt='%Y-%m-%d %H:%M:%S')

    def getRandomNumber(self):
        ability = str(random.randint(1, 10))
        self.updateDisplay(ability)

    def updateDisplay(self, myString):
        self.displayVariable.set(myString)

    def manipulateFont(self, fontSize=None, *args):
        newFont = (font.get(), fontSize.get())
        return newFont

    def clear(self):
        txt.delete(0, 'end')
        res = ""
        self.message.configure(text=res)

    def clear2(self, txt2=None):
        txt2.delete(0, 'end')
        res = ""
        self.message.configure(text=res)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def collectUserImageForRegistration(self):
        clientIDVal = (self.clientIDTxt.get())
        empIDVal = self.empIDTxt.get()
        name = (self.empNameTxt.get())
        ap = argparse.ArgumentParser()

        ap.add_argument("--faces", default=50,
                        help="Number of faces that camera will get")
        ap.add_argument("--output", default="../datasets/train/" + name,
                        help="Path to faces output")

        args = vars(ap.parse_args())

        trnngDataCollctrObj = TrainingDataCollector(args)
        trnngDataCollctrObj.collectImagesFromCamera()

        notifctn = "We have collected " + str(args["faces"]) + " images for training."
        self.message.configure(text=notifctn)

    def getFaceEmbedding(self):

        ap = argparse.ArgumentParser()

        ap.add_argument("--dataset", default="../datasets/train",
                        help="Path to training dataset")
        ap.add_argument("--embeddings", default="faceEmbeddingModels/embeddings.pickle")
        # Argument of insightface
        ap.add_argument('--image-size', default='112,112', help='')
        ap.add_argument('--model', default='../insightface/models/model-y1-test2/model,0', help='path to load model.')
        ap.add_argument('--ga-model', default='', help='path to load model.')
        ap.add_argument('--gpu', default=0, type=int, help='gpu id')
        ap.add_argument('--det', default=0, type=int,
                        help='mtcnn option, 1 means using R+O, 0 means detect from begining')
        ap.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
        ap.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
        args = ap.parse_args()

        genFaceEmbdng = GenerateFaceEmbedding(args)
        genFaceEmbdng.genFaceEmbedding()

    def trainModel(self):
        # ============================================= Training Params ====================================================== #

        ap = argparse.ArgumentParser()

        # ap = argparse.ArgumentParser()
        ap.add_argument("--embeddings", default="faceEmbeddingModels/embeddings.pickle",
                        help="path to serialized db of facial embeddings")
        ap.add_argument("--model", default="faceEmbeddingModels/my_model.h5",
                        help="path to output trained model")
        ap.add_argument("--le", default="faceEmbeddingModels/le.pickle",
                        help="path to output label encoder")

        args = vars(ap.parse_args())

        self.getFaceEmbedding()
        faceRecogModel = TrainFaceRecogModel(args)
        faceRecogModel.trainKerasModelForFaceRecognition()

        notifctn = "Model training is successful.No you can go for prediction."
        self.message.configure(text=notifctn)

    def makePrediction(self):
        faceDetector = FacePredictor()
        faceDetector.detectFace()

    def close_window(self):
        self.window.destroy()

    def callback(self, url):
        webbrowser.open_new(url)


logFileName = "ProceduralLog.txt"
regStrtnModule = RegistrationModule(logFileName)
# regStrtnModule = RegistrationModule
# regStrtnModule.TrainImages()