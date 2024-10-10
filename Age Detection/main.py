import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load age and gender detection models
age_weights = "C:\\oops\\PyVerse\\Deep_Learning\\Object detection\\age_net.caffemodel"
age_config = "C:\\oops\\PyVerse\\Deep_Learning\\Object detection\\age_deploy.prototxt"
gender_weights = "C:\\oops\\PyVerse\\Deep_Learning\\Object detection\\gender_net.caffemodel"
gender_config = "C:\\oops\\PyVerse\\Deep_Learning\\Object detection\\gender_deploy.prototxt"

# Load pre-trained models for age and gender detection
age_net = cv2.dnn.readNet(age_config, age_weights)
gender_net = cv2.dnn.readNet(gender_config, gender_weights)

# List of age ranges and genders
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
           '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']
model_mean = (78.4263377603, 87.7689143744, 114.895847746)

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_predict_age_gender(frame):
    try:
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

        if len(faces) == 0:
            print("No face detected.")
        else:
            for (x, y, w, h) in faces:
                box = [x, y, x + w, y + h]
                face = frame[box[1]:box[3], box[0]:box[2]]

                # Preprocess face for model input
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean, swapRB=False)

                # Age Prediction
                age_net.setInput(blob)
                age_preds = age_net.forward()
                age = age_list[age_preds[0].argmax()]

                # Gender Prediction
                gender_net.setInput(blob)
                gender_preds = gender_net.forward()
                gender = gender_list[gender_preds[0].argmax()]

                # Draw bounding box and label
                label = f'{gender}, {age}'
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        return frame

    except Exception as e:
        print(f"Error during detection: {e}")
        return frame

# Real-time age and gender detection using webcam
def real_time_detection():
    cap = cv2.VideoCapture(0)  # Start webcam
    if not cap.isOpened():
        print("Error: Unable to access the camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera")
            break

        # Perform detection and predictions
        result_frame = detect_and_predict_age_gender(frame)

        # Display result
        cv2.imshow('Age and Gender Detection', result_frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Save processed image
def save_image(image, output_path="output_image.jpg"):
    try:
        cv2.imwrite(output_path, image)
        print(f"Image saved at {output_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

if __name__ == "__main__":
    # Real-time detection
    real_time_detection()

    # For static image, uncomment the following:
    # img = cv2.imread('C:\\oops\\PyVerse\\Deep_Learning\\Object detection\\OIP.jpg')
    # img = cv2.resize(img, (720, 640))
    # result_img = detect_and_predict_age_gender(img)
    # plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()

    # Option to save image
    # save_image(result_img, 'detected_output.jpg')
