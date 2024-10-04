import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def load_yolo_model(weights_path, cfg_path):
    return cv2.dnn.readNet(weights_path, cfg_path)

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Error loading image.")
    return img

def perform_detection(yolo, img):
    height, width = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    yolo.setInput(blob)
    return yolo.forward(yolo.getUnconnectedOutLayersNames()), height, width

def draw_detections(img, layer_output, height, width, classes):
    boxes, confidences, class_ids = [], [], []
    
    for output in layer_output:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    if len(indexes) > 0:
        indexes = indexes.flatten()
        for i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = np.random.randint(0, 255, size=3).tolist()
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

def main():
    weights_path = os.path.join("Dataset", "yolov3.weights")
    cfg_path = os.path.join("Dataset", "yolov3.cfg")
    yolo = load_yolo_model(weights_path, cfg_path)

    with open(os.path.join("Dataset", "coco.names"), 'r') as f:
        classes = f.read().splitlines()

    img = preprocess_image("C:\\Users\\billa\\Desktop\\Programs\\ML_DL\\gg.png")
    layer_output, height, width = perform_detection(yolo, img)

    draw_detections(img, layer_output, height, width, classes)

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
