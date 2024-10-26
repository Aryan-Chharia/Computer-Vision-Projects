# Real-time Sketch Effect using OpenCV

This project applies a real-time, detailed sketch effect to video feed from a webcam using OpenCV and Python. The program captures frames from the webcam, processes them using both Canny Edge Detection and Laplacian filters to generate a more detailed sketch effect, and displays the result in real-time.

## Features

- **Real-time video capture** from the webcam.
- **Edge detection** using both Canny and Laplacian filters for enhanced sketch detailing.
- **Customizable parameters** for Gaussian blurring, edge detection thresholds, and binary inversion.
- User-friendly interface that allows quitting the program by pressing the 'q' key.

## Requirements

To run this program, ensure you have the following installed:

- Python 3.x
- OpenCV (`cv2`) Python library
- NumPy Python library

### Install Dependencies

If you do not have OpenCV and NumPy installed, you can install them via pip:

```bash
pip install opencv-python numpy
```

## How to Run

1. Clone the repository or download the project files.

2. Ensure that your system has a working webcam.

3. Navigate to the project folder and run the Python script:

```bash
python sketch_effect.py
```

4. A window will open showing the real-time detailed sketch effect applied to your webcam feed.

5. Press `q` to quit the program.

## Code Explanation

The main function of this project is the `sketch()` function, which applies the following steps to create the detailed sketch effect:

1. **Grayscale Conversion**: Converts the video frame into grayscale.
2. **Gaussian Blur**: Applies a blur to smooth the image while retaining edges.
3. **Edge Detection**: Uses both Canny Edge Detection and Laplacian filters to detect and combine edges.
4. **Binary Thresholding**: Converts the detected edges into a binary image for a detailed sketch-like effect.

The processed video frames are displayed in real-time, and the program exits when the user presses the `q` key.

### Code Snippet: Sketch Function

```python
def sketch(frame, blur_kernel=(3, 3), canny_threshold1=30, canny_threshold2=100, threshold_value=150):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, blur_kernel, 0)
    edges = cv2.Canny(gray_blur, canny_threshold1, canny_threshold2)
    laplacian = cv2.Laplacian(gray_blur, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    combined_edges = cv2.add(edges, laplacian)
    ret, mask = cv2.threshold(combined_edges, threshold_value, 255, cv2.THRESH_BINARY_INV)
    return mask
```

### Parameters

- **`blur_kernel`**: Kernel size for Gaussian blur (default is `(3, 3)`).
- **`canny_threshold1`**: Lower threshold for Canny edge detection (default is `30`).
- **`canny_threshold2`**: Upper threshold for Canny edge detection (default is `100`).
- **`threshold_value`**: Threshold for binary inversion (default is `150`).

## Customization

You can easily adjust the parameters in the `sketch()` function to fine-tune the effect:

- Change the **kernel size** of the Gaussian blur for different levels of smoothness.
- Adjust the **Canny edge detection thresholds** for different sensitivity to edges.
- Modify the **binary threshold** to control how the sketch is rendered.

## Example Output

This program will open two windows:
- One showing the live webcam feed.
- Another displaying the live detailed sketch effect applied to the feed.

Press 'q' to quit and close both windows.


**Note**: If you encounter any issues with the webcam not opening, please check if your system has a functional webcam and that OpenCV is installed properly.