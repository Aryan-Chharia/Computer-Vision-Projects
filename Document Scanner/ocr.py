import pytesseract
from PIL import Image
import cv2
import numpy as np

# Specify the correct path to Tesseract-OCR
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    """
    Preprocess the image for better OCR results.
    - Convert to grayscale
    - Apply Gaussian blur
    - Perform adaptive thresholding
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 11, 2)
    return thresh_image

def extract_text(image, lang='eng'):
    """
    Extract text from the provided image using OCR.
    
    :param image: Input image (OpenCV format).
    :param lang: Language for OCR (default is English).
    :return: Extracted text as a string.
    """
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)

        # Convert the image from OpenCV format to PIL format for pytesseract
        pil_image = Image.fromarray(processed_image)

        # Perform OCR to extract text
        text = pytesseract.image_to_string(pil_image, lang=lang)
        return text
    except Exception as e:
        print(f"Error during OCR: {e}")
        return ""

# Example usage (optional)
if __name__ == "__main__":
    # Load an image using OpenCV
    input_image = cv2.imread('path_to_image.jpg')

    # Extract text from the image
    extracted_text = extract_text(input_image)

    # Print the extracted text
    print("Extracted Text:\n", extracted_text)
