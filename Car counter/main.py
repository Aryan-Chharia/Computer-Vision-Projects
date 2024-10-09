import cv2
import numpy as np
import os

# Capture video from camera index 2
cap = cv2.VideoCapture(2)

# Background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Define counting line
count_line_position = 300
offset = 6  # Allowable error between pixels

# Initialize car count
car_count = 0

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')

# Calculate the center of a rectangle (used for object tracking)
def get_center(x, y, w, h):
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy

detected_cars = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    
    # Remove noise and shadows
    _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Detect contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Ignore small objects
        if cv2.contourArea(cnt) < 2000:
            continue

        # Get the bounding box for the contour
        x, y, w, h = cv2.boundingRect(cnt)

        # Get the center of the detected object
        center = get_center(x, y, w, h)

        # Draw rectangle around the detected vehicle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Add the center point to the list of detected cars
        detected_cars.append(center)

    # Draw the counting line
    cv2.line(frame, (0, count_line_position), (640, count_line_position), (0, 0, 255), 2)

    # Count cars crossing the line
    for (x, y) in detected_cars:
        if count_line_position - offset < y < count_line_position + offset:
            car_count += 1
            detected_cars.remove((x, y))
            
            # Write results to HTML file
            with open('results/results.html', 'w') as f:
                f.write(f"""
                <html>
                <head><title>Car Count Results</title></head>
                <body>
                    <h1>Car Count Results</h1>
                    <p>Total Cars Counted: {car_count}</p>
                </body>
                </html>
                """)

    # Display car count on frame
    cv2.putText(frame, "Car Count: " + str(car_count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the frame
    cv2.imshow('Car Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
