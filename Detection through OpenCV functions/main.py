import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to display images using matplotlib
def show_image(title, img):
    # Convert BGR image to RGB for displaying in matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Function for Color Detection
def detect_color(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_color = np.array([35, 100, 100])  # Example for green color
    upper_color = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    result = cv2.bitwise_and(img, img, mask=mask)
    return result

# Function for Shape Detection
def detect_shapes(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        cv2.drawContours(img, [approx], 0, (0, 255, 0), 3)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 10
        if len(approx) == 3:
            cv2.putText(img, "Triangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        elif len(approx) == 4:
            cv2.putText(img, "Rectangle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        elif len(approx) == 5:
            cv2.putText(img, "Pentagon", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        else:
            cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return img

# Function for Edge Detection
def detect_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

# Function for Image Filtering
def apply_filters(img):
    blur = cv2.GaussianBlur(img, (15, 15), 0)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpen = cv2.filter2D(img, -1, sharpen_kernel)
    return blur, sharpen

# Main function to apply all processes
def process_image(image_path):
    img = cv2.imread(image_path)

    # Perform Color Detection
    color_detected = detect_color(img.copy())
    show_image("Color Detection", color_detected)

    # Perform Shape Detection
    shapes_detected = detect_shapes(img.copy())
    show_image("Shape Detection", shapes_detected)

    # Perform Edge Detection
    edges_detected = detect_edges(img.copy())
    plt.imshow(edges_detected, cmap='gray')  # Display grayscale edge detection
    plt.title("Edge Detection")
    plt.axis('off')
    plt.show()

    # Perform Image Filtering (blur and sharpen)
    blurred, sharpened = apply_filters(img.copy())
    show_image("Blurred Image", blurred)
    show_image("Sharpened Image", sharpened)

# Use any image path for testing
image_path = 'C:\\Users\\billa\\OneDrive\\Documents\\ABC\\Computer-Vision-Projects\\Detection through OpenCV functions\\Results\\original.jpg'
process_image(image_path)
