# YOLOv3 Object Detection with OpenCV

# Description
Object detection is a fundamental aspect of computer vision, allowing us to identify and classify objects in images or videos. It helps pinpoint where objects are and what categories they belong to. This project uses the YOLOv3 (You Only Look Once) model which has a great balance between speed and accuracy, making it ideal for real-time applications. With the help of OpenCV, this project processes images, applies the YOLOv3 model, and displays the detected objects with labeled bounding boxes.

# Features
- Real-Time Object Detection: Detects various objects in images using the YOLOv3 model.
- Visualization: Draws bounding boxes around detected objects and labels them with class names like car, person, bicycle, etc.
- Adjustable Confidence: Allows adjustment of the detection confidence threshold, controlling sensitivity to detected objects.
- Non-Maximum Suppression (NMS): Uses NMS to eliminate overlapping bounding boxes, ensuring more accurate detections.
- Flexibility: Easy to customize for different image inputs and parameters.

# Requirements
- Python 3.x
- opencv-python
- numpy
- matplotlib

### To install the required packages:
```pip install opencv-python numpy matplotlib```

# How the project works:
- **Loading the YOLO Model**: The ```load_yolo_model``` function sets up the YOLO model by loading its configuration (yolov3.cfg) and weight files (yolov3.weights). These files define how the model is built and the knowledge it has learned.

- **Preprocessing the Image**: The ```preprocess_image``` function reads the input image with OpenCV and preps it for YOLO. This involves converting the image into a "blob" format, which means resizing it and adjusting the pixel values to suit the model's needs.

- **Performing Object Detection**: The ```perform_detection``` function takes the prepped image and runs it through the YOLO model. This step is where the model figures out what objects are in the image and where they are. It gathers the locations and probabilities of each detected object.
 
- **Drawing Detections**: The ```draw_detections``` function sorts through the model’s predictions, keeping only those that meet a certain confidence level. It then draws boxes around the detected objects and adds labels using OpenCV’s drawing tools to show what each object is.

- **Displaying the Result**: The final detected image is displayed using matplotlib. This lets you see what objects the model found and their locations in the image in a pop-up window.

# Input
![This is an alt text.](https://github.com/Aryan-Chharia/Computer-Vision-Projects/blob/main/Object%20detection/Result/Input.png?raw=true "This is a sample image.")

# Output
![This is an alt text.](https://github.com/Aryan-Chharia/Computer-Vision-Projects/blob/main/Object%20detection/Result/Output.png?raw=true "This is a sample image.")

# Conclusion
This project showcases how YOLOv3 can be used for real-time object detection, making it a great tool for a variety of practical applications. YOLOv3 strikes a good balance between speed and accuracy, which makes it suitable for tasks that need quick responses without sacrificing performance. This setup serves as a solid foundation for building custom object detection systems, whether for surveillance, self-driving cars, or other similar needs.
