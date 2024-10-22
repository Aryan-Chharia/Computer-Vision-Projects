## ✔ REAL TIME HUMAN DETECTION & COUNTING

- A tensorflow based `Faster RCNN inception v2` python model to detect and count humans in real time images, videos & camera.
- Used pre-trained `frozen_inference_graph.pb` frozen graph to handle the detection.
- Visualize the data using `Enumeration Plot` and `Avg. Accuracy Plot`.

---

### 📌REQUIREMENTS :

- python 3
- tkinter
- messagebox
- PIL
- cv2
- argparse
- matplotlib.pyplot
- numpy
- time
- os
- tensorflow
- fpdf

---

### 📌How this Script works :

- User just need to download the file and run the main.py on their local system.
- On the starting window of the application, user will be able to see START and EXIT option, using which user can start the application or exit from the application.
- When user starts the application using START button, a new window will open, which allows user with options like, DETECT FROM IMAGE, DETECT FROM VIDEO or DETECT FROM CAMERA.
- When user selects any of the first two option, he/she needs to select the respective files using SELECT button.
- User can preview the selected file using PREVIEW button, and detect and count the humans using DETECT button.
- And when user selects, the last option of detecting through camera, user need to open the Camera, using OPEN CAMERA button, As soon as camera opens, detection process will start.
- After detection process gets completed or user manually completes it, two graph get plotted,
  - 1.) Enumeration Plot(Human Count Vs. time) and
  - 2.) Avg. Accuracy Plot(Avg. Accuracy Vs. time).
- Along with this two plots, an option to generate crowd report also appears, On clicking on it, a crowd report in form of PDF is generated ans saved autmatically at the project file location.
- In the crowd report genrated, there will be information like, What is Max Human Count, Max Accuracy, Max Avg. Accuracy, and also a two line status about crowd.

### 📌Purrpose :

- This scripts helps user to easily get the count of human through real time image, video or camera, and thereafter also analysis of crowd through crowd report.

### 📌Compilation Steps :

- Install all the required libraries.
- After that download the code file, and run main.py on local system.
- Then the script will start running and user can explore it by detecting the human and also getting the count of it.

---

### 📌SCREENSHOTS :

<p align="center">
  <img width = 1000 src="Screenshots/1.jpg" /><br>
  <img width = 1000 src="Screenshots/2.jpg" /><br>
</p>

- #### Image:

<p align="center">
  <img width = 1000 src="Screenshots/3.jpg" /><br>
  <img width = 1000 src="Screenshots/4.jpg" /><br>
  <img width = 1000 src="Screenshots/5.jpg" /><br>
  <img width = 1000 src="Screenshots/6.jpg" /><br>
  <img width = 1000 src="Screenshots/7.jpg" /><br>
  <img width = 1000 src="Screenshots/8.jpg" /><br>
  <img width = 1000 src="Screenshots/9.jpg" /><br>
  <img width = 1000 src="Screenshots/10.jpg" /><br>
  <img width = 1000 src="Screenshots/11.jpg" /><br>
  <img width = 1000 src="Screenshots/12.jpg" /><br>
</p>

- #### Video:

<p align="center">
  <img width = 1000 src="Screenshots/13.jpg" /><br>
  <img width = 1000 src="Screenshots/14.jpg" /><br>
  <img width = 1000 src="Screenshots/15.jpg" /><br>
  <img width = 1000 src="Screenshots/16.jpg" /><br>
  <img width = 1000 src="Screenshots/17.jpg" /><br>
</p>

- #### Camera:

<p align="center">
  <img width = 1000 src="Screenshots/18.jpg" /><br>
  <img width = 1000 src="Screenshots/19.jpg" /><br>
  <img width = 1000 src="Screenshots/20.jpg" /><br>
  <img width = 1000 src="Screenshots/21.jpg" /><br>
  <img width = 1000 src="Screenshots/22.jpg" /><br>
  <img width = 1000 src="Screenshots/23.jpg" /><br>
</p>

---
