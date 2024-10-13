# Age Detection Using OpenCV and Caffe
This project implements an age detection system using OpenCV's DNN module and a pre-trained Caffe model. The system detects faces in an image and predicts the age group of each detected face.

## Table of Contents
**Requirements
Installation
Usage
Project Structure
Models
Results
Troubleshooting**

### Requirements
To run this project, you will need:

Python 3.7+
OpenCV 4.5+
Numpy
Matplotlib
Model Files

The following files are required to run the model:

Pre-trained age detection model (Caffe):
age_deploy.prototxt: Network configuration file.
mobilenet_iter_73000.caffemodel: Pre-trained model weights.
Haar Cascade classifier for face detection:
haarcascade_frontalface_default.xml: OpenCV's Haar Cascade classifier for face detection.
You can download the necessary files from the links provided in the Models section.

### Installation
Clone the repository:

```python

Copy code
git clone https://github.com/your-repo/age-detection-opencv.git
cd age-detection-opencv
```
Install dependencies:

Install the required Python packages using pip:

```python

Copy code
pip install numpy opencv-python matplotlib
```
### Usage
Prepare the image:

Place the image you want to process in the project directory. Ensure that the path in the script points to this image file.

Run the script:

To run the age detection script, use the following command:

```python

Copy code
python age.py
```
The script will detect faces in the image and display the predicted age for each detected face.

Expected Output:

The image will be displayed with bounding boxes around the detected faces.
Predicted age ranges (e.g., (25-32)) will be shown for each detected face.
Project Structure
```python

Copy code
.
├── age.py                            # Main script to run age detection
├── mobilenet_iter_73000.caffemodel    # Pre-trained Caffe model (downloaded)
├── age_deploy.prototxt                # Caffe model configuration file (downloaded)
├── haarcascade_frontalface_default.xml# Haar Cascade face detector
├── OIP.jpg                            # Sample image
└── README.md                          # This file
```
Models
Download the pre-trained models from this dataset mentioned:

Age detection model:

age_deploy.prototxt
mobilenet_iter_73000.caffemodel
Face detection model (Haar Cascade): haarcascade_frontalface_default.xml

### Results
The model will output an image with detected faces and the predicted age range for each face, such as:

Face 1: (25-32)
Face 2: (15-20)
The results are displayed using Matplotlib and OpenCV.

### Troubleshooting
Common Errors
cv2.error: (-215:Assertion failed) when loading Haar Cascade:

Ensure the path to haarcascade_frontalface_default.xml is correct. Update the path in the script if necessary.

Number of input channels should be multiple of 32 error:

This error usually occurs due to a mismatch between the input image and the model’s expected input size. Ensure that the image is properly resized and preprocessed before feeding it to the model.

Use the following blob creation for preprocessing the face image:

```python

Copy code
blob = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(224, 224), mean=model_mean, swapRB=False, crop=False)

```
No face detected: Ensure the input image is clear and well-lit for better face detection results. Increase the number of neighbors or adjust scaleFactor in detectMultiScale if necessary.


### Instructions for Usage:
1. **Save the code above** as `age.py`.
2. Ensure that all necessary files (`mobilenet_iter_73000.caffemodel`, `age_deploy.prototxt`, `haarcascade_frontalface_default.xml`, and your test image `OIP.jpg`) are in the same directory or update the paths in the code accordingly.
3. Install the required packages as specified in the README.
4. Run the script using the command:
   ```bash
   python age.py

# Load models
age_weights = "C:\\oops\\PyVerse\\Deep_Learning\\Object detection\\age_net.caffemodel"
age_config = "C:\\oops\\PyVerse\\Deep_Learning\\Object detection\\age_deploy.prototxt"
age_net = load_age_model(age_config, age_weights)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces
faces = detect_faces(frame, face_cascade)

if len(faces) == 0:
    message = 'No face detected'
    cv2.putText(img, message, (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow('Output', img)
    cv2.waitKey(0)
else:
    for (x, y, w, h) in faces:
        box = [x, y, x + w, y + h]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the face from the frame
        face = frame[box[1]:box[3], box[0]:box[2]]

        # Predict age
        age = predict_age(face, age_net, (78.4263377603, 87.7689143744, 114.895847746))

        # Display the age on the frame
        cv2.putText(frame, f'Age: {age}', (box[0], box[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the processed frame
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    cv2.imshow('Output', frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()

"""
# Age Detection Using OpenCV and Caffe

This project implements an age detection system using OpenCV's DNN module and a pre-trained Caffe model.
The system detects faces in an image and predicts the age group of each detected face.

## Table of Contents
- Requirements
- Installation
- Usage
- Project Structure
- Models
- Results
- Troubleshooting

### Requirements
To run this project, you will need:
- Python 3.7+
- OpenCV 4.5+
- NumPy
- Matplotlib

#### Model Files
The following files are required to run the model:
- Pre-trained age detection model (Caffe):
  - `age_deploy.prototxt`: Network configuration file.
  - `mobilenet_iter_73000.caffemodel`: Pre-trained model weights.
- Haar Cascade classifier for face detection:
  - `haarcascade_frontalface_default.xml`: OpenCV's Haar Cascade classifier for face detection.

You can download the necessary files from the links provided in the Models section.

### Installation
Clone the repository:

```bash
git clone https://github.com/your-repo/age-detection-opencv.git
cd age-detection-opencv
