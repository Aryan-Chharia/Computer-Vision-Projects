import cv2
import numpy as np
from utils import order_points, four_point_transform

def scan_document(image_path):
    # Load the image and convert it to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Use GaussianBlur and edge detection (Canny)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)

    # Find contours and sort by area, descending
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    # Loop over contours to find a quadrilateral
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(approx) == 4:
            # If a 4-point contour is found, perform perspective correction
            doc_contour = approx
            break

    # Apply the perspective transformation
    warped = four_point_transform(image, doc_contour.reshape(4, 2))
    
    # Convert the warped image to grayscale and threshold for better clarity
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, scanned_image = cv2.threshold(warped_gray, 127, 255, cv2.THRESH_BINARY)
    
    return scanned_image
