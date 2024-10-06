import pytesseract
from PIL import Image
import cv2

# Specify the correct path to Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(image):
    # Convert the image from OpenCV format to PIL format for pytesseract
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Perform OCR to extract text
    text = pytesseract.image_to_string(pil_image)
    return text
