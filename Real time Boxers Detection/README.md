#Boxer Detection and Tracking using YOLO and OpenCV

This project is focused on detecting and tracking boxers from a video using the YOLO (You Only Look Once) object detection algorithm and OpenCV. By analyzing the video, we extract positional information of the boxers and save it in a .npy format for further processing.

Steps to Follow

Step 1: Download the Video Start by downloading the video file containing the boxing match. Ensure the video is in a compatible format such as .mp4 or .avi. The video will be used as input for detection and tracking of the boxers.

Step 2: Install Necessary Packages Before running the code, install the required packages and dependencies. You can do this by creating a virtual environment and installing the necessary libraries.

The primary libraries needed are:

opencv-python

numpy

yolo-v5 (or appropriate YOLO model version)

torch (if using PyTorch for YOLO)

To install these packages, use:

pip install opencv-python numpy torch

pip install yolo-v5

Step 3: Detect and Track the Boxers using YOLO and OpenCV Utilize YOLO for object detection in combination with OpenCV to process the video. YOLO is a fast object detection algorithm that can accurately identify the boxers in the frames of the video. OpenCV will be used for video frame manipulation and to apply the YOLO model.

Load the video.

Apply YOLO to each frame to detect boxers.

Track the boxers by identifying their bounding boxes across the frames.

Step 4: Get Positional Information of Each Boxer Once the boxers are detected, extract their positional information (bounding box coordinates: x, y, width, height) in each frame. This data can be used to track their movements across the video.

Step 5: Save the Positional Information in .npy Format The positional information obtained from the video will be stored in a NumPy array and saved in .npy format for further analysis. This can be done using the numpy.save() function.

Example to save positional information
positional_info = [] # Store positional data of boxers

np.save('boxer_positions.npy', np.array(positional_info))

This will generate a .npy file that contains the positional data of the boxers over time, which can be used in further applications or analysis.
