from flask import Flask, render_template, request
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def preprocess_signature(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to a standard size
    resized_image = cv2.resize(gray_image, (128, 128))
    # Threshold the image (binary)
    _, binary_image = cv2.threshold(resized_image, 128, 255, cv2.THRESH_BINARY_INV)
    return binary_image

def compute_histogram(image):
    # Compute the histogram (256 bins for grayscale image)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def compare_signatures(signature1, signature2):
    # Compute histograms for both signatures
    hist1 = compute_histogram(signature1)
    hist2 = compute_histogram(signature2)
    # Compare histograms using correlation method
    score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    if 'signature1' not in request.files or 'signature2' not in request.files:
        return "No files uploaded", 400

    file1 = request.files['signature1']
    file2 = request.files['signature2']
    
    path1 = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
    path2 = os.path.join(app.config['UPLOAD_FOLDER'], file2.filename)
    
    file1.save(path1)
    file2.save(path2)

    # Process the signatures
    sig1 = preprocess_signature(path1)
    sig2 = preprocess_signature(path2)

    # Compare signatures
    similarity_score = compare_signatures(sig1, sig2)

    # Set a threshold for verification
    threshold = 0.7
    result = "<b>The signatures matched successfully.</b>" if similarity_score > threshold else "<b>The signatures are likely to be different.</b>"
    
    return result

if __name__ == '__main__':
    app.run(debug=True)
