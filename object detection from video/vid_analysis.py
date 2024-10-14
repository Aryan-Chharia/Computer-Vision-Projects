import cv2
import numpy as np
import os

def analyze_video(video_path, output_video_path='output_video.avi', confidence_threshold=0.5, nms_threshold=0.4, process_every_nth_frame=1):
    # Replace with actual paths
    cfg_path = "video/yolov3.cfg"
    weights_path = "video/yolov3.weights"
    names_path = "video/coco.names"

    # Check if paths exist
    if not all(map(os.path.exists, [cfg_path, weights_path, names_path, video_path])):
        print("Error: One or more file paths are incorrect.")
        return

    # Load YOLO
    net = cv2.dnn.readNet(weights_path, cfg_path)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Load class names
    with open(names_path, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0
    image_save_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every nth frame
        if frame_count % process_every_nth_frame == 0:
            # Create a blob from the current frame
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # Analyze outputs to find detected objects
            class_ids = []
            confidences = []
            boxes = []
            for out_data in outs:
                for detection in out_data:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > confidence_threshold:
                        # Object detected, get the bounding box parameters
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Apply non-maxima suppression to remove redundant boxes
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
            descriptions = []
            if len(indexes) > 0:
                for i in indexes.flatten():
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    confidence = confidences[i]
                    # Draw a green rectangle around the detected object
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # Display the label and confidence
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    descriptions.append(f"{label} ({confidence:.2f})")

            # Print detected objects description to the console
            description_text = "Detected objects: " + ", ".join(descriptions)
            print(description_text)

            # Write the processed frame to the output video
            out.write(frame)

            # Save the frame as an image file
            image_filename = f'detected_frame_{image_save_count}.jpg'
            cv2.imwrite(image_filename, frame)
            image_save_count += 1

        frame_count += 1

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Processed video saved as {output_video_path}")

if __name__ == '__main__':
    video_path = 'D:/proj4/video/main_vid.mp4'
    output_video_path = 'D:/proj4/output/output_video.avi'
    analyze_video(video_path, output_video_path, confidence_threshold=0.5, nms_threshold=0.4, process_every_nth_frame=1)
