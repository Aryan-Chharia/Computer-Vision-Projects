import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to load the age detection model
def load_age_model(age_config_path, age_weights_path):
    age_net = cv2.dnn.readNet(age_config_path, age_weights_path)
    return age_net

# Function to detect faces in an image
def detect_faces(image, face_cascade):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
    return faces

# Function to predict age from a detected face
def predict_age(face, age_net, model_mean):
    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), model_mean, swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', 
                '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    age = age_list[age_preds[0].argmax()]
    return age

# Function to save the output image
def save_output_image(frame, output_path):
    cv2.imwrite(output_path, frame)
    print(f"Output image saved at {output_path}")

# Main function for processing the image
def process_image(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to load image.")
        return
    img = cv2.resize(img, (720, 640))
    frame = img.copy()

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

        # Show the output window with OpenCV
        cv2.imshow('Output', frame)
        cv2.waitKey(0)

        # Save the output image
        output_path = os.path.join(os.path.dirname(image_path), "output.jpg")
        save_output_image(frame, output_path)

    cv2.destroyAllWindows()

# Function for real-time detection using webcam
def real_time_detection():
    age_weights = "C:\\oops\\PyVerse\\Deep_Learning\\Object detection\\age_net.caffemodel"
    age_config = "C:\\oops\\PyVerse\\Deep_Learning\\Object detection\\age_deploy.prototxt"
    age_net = load_age_model(age_config, age_weights)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to access webcam.")
            break

        faces = detect_faces(frame, face_cascade)

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                box = [x, y, x + w, y + h]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                face = frame[box[1]:box[3], box[0]:box[2]]
                age = predict_age(face, age_net, (78.4263377603, 87.7689143744, 114.895847746))
                cv2.putText(frame, f'Age: {age}', (box[0], box[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Real-Time Age Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Execute the main function or real-time detection based on user input
if __name__ == "__main__":
    choice = input("Choose an option:\n1. Process Image\n2. Real-Time Detection\nEnter choice (1 or 2): ")

    if choice == '1':
        image_path = input("Enter the image path: ")
        process_image(image_path)
    elif choice == '2':
        real_time_detection()
    else:
        print("Invalid choice. Please select 1 or 2.")
