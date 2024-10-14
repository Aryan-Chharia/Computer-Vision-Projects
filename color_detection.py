import cv2
import numpy as np

# Define a function to get the color name based on BGR values
def get_color_name(bgr):
    # Define a dictionary of color names and their BGR values
    colors = {
        "Red": [0, 0, 255],
        "Green": [0, 255, 0],
        "Blue": [255, 0, 0],
        "Yellow": [0, 255, 255],
        "Cyan": [255, 255, 0],
        "Magenta": [255, 0, 255],
        "Black": [0, 0, 0],
        "White": [255, 255, 255],
        "Gray": [128, 128, 128]
    }

    # Find the closest color
    min_dist = float('inf')
    color_name = "Unknown"
    for name, value in colors.items():
        dist = np.linalg.norm(np.array(bgr) - np.array(value))
        if dist < min_dist:
            min_dist = dist
            color_name = name

    return color_name

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to RGB (OpenCV uses BGR by default)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the height and width of the frame
    height, width, _ = rgb_frame.shape

    # Define a region of interest (ROI) for color detection
    roi = rgb_frame[int(height/2)-50:int(height/2)+50, int(width/2)-50:int(width/2)+50]

    # Calculate the average color in the ROI
    average_color = cv2.mean(roi)[:3]  # Get the average BGR values

    # Get the color name
    color_name = get_color_name(average_color)

    # Display the color name on the frame
    cv2.putText(frame, f"Color: {color_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow("Color Detection", frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
