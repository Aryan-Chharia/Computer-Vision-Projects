import cv2
import numpy as np
import pytesseract
from pytesseract import Output

# Initialize Tesseract path (for Windows, adjust the path as needed)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to preprocess image for better OCR results
def preprocess_image(img):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve OCR accuracy
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive threshold to highlight text/sketches
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    return thresh

# Function to extract text from image
def extract_text(img):
    # Use Tesseract OCR to extract text
    custom_config = r'--oem 3 --psm 6'  # Optimal config for capturing single-line text
    text = pytesseract.image_to_string(img, config=custom_config)
    return text

# Function to extract bounding boxes for words detected
def extract_text_with_boxes(img):
    d = pytesseract.image_to_data(img, output_type=Output.DICT)
    boxes = []
    n_boxes = len(d['level'])
    for i in range(n_boxes):
        if int(d['conf'][i]) > 60:  # Confidence threshold to filter low-quality detections
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            boxes.append((x, y, w, h, d['text'][i]))
    return boxes

# Real-time capture and processing loop
def capture_whiteboard():
    cap = cv2.VideoCapture(0)  # Use default camera, or specify another index
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        processed_img = preprocess_image(frame)
        
        # Extract text
        text = extract_text(processed_img)
        print("Extracted Text:", text)

        # Extract and draw bounding boxes around detected text
        boxes = extract_text_with_boxes(processed_img)
        for (x, y, w, h, txt) in boxes:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, txt, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show original frame with bounding boxes
        cv2.imshow("Digital Whiteboard Reader", frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the capture function
capture_whiteboard()
