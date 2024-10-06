import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO("yolov8/yolov8n-face.pt")

# Start video capture
video_capture = cv2.VideoCapture(0)

# Load and encode images
image_dir = "photos/"
encodings = {}
known_face_names = []

for fname in os.listdir(image_dir):
    file_path = os.path.join(image_dir, fname)
    image = face_recognition.load_image_file(file_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encoding = face_recognition.face_encodings(img)
    if encoding:
        encodings[fname] = encoding[0]
        known_face_names.append(fname.split('.')[0])  # Assuming the names are derived from file names

# Known face encodings and names
known_face_encodings = list(encodings.values())
known_face_names = [name.split('.')[0] for name in encodings.keys()]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s = True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

# Create or open a CSV file for the current date
with open(current_date + '.csv', 'w+', newline='') as f:
    lnwriter = csv.writer(f)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Get YOLO results
        yolo_results = yolo_model(rgb_small_frame)
        yolo_boxes = yolo_results[0].boxes

        # Draw rectangles for YOLO-detected faces
        for box in yolo_boxes:
            top_left_x, top_left_y = int(box.xyxy.tolist()[0][0]), int(box.xyxy.tolist()[0][1])
            bottom_right_x, bottom_right_y = int(box.xyxy.tolist()[0][2]), int(box.xyxy.tolist()[0][3])
            cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (50, 200, 129), 2)

        if s:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            if face_locations:
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = ""
                    face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distance)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

                    # Display the name on the frame
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10, 50)
                    fontScale = 1.5
                    fontColor = (0, 255, 0)
                    thickness = 3
                    lineType = 2

                    if name in known_face_names:
                        cv2.putText(frame, name + ' present', 
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)
                        
                        if name in students:
                            students.remove(name)
                            print(students)
                            lnwriter.writerow([name, 'present'])
                    else:
                        cv2.putText(frame, 'unknown',
                            bottomLeftCornerOfText, 
                            font, 
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Write absent students to CSV
    for name in students:
        lnwriter.writerow([name, 'absent'])

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
