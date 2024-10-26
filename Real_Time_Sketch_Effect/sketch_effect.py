import cv2
import numpy as np

def sketch(frame, blur_kernel=(3, 3), canny_threshold1=30, canny_threshold2=100, threshold_value=150):
    """
    Generate a more detailed sketch effect from a given frame.

    Parameters:
    - frame: The input image frame from the webcam.
    - blur_kernel: The kernel size for Gaussian blurring (default is (3, 3)).
    - canny_threshold1: The lower threshold for the Canny edge detector (default is 30).
    - canny_threshold2: The upper threshold for the Canny edge detector (default is 100).
    - threshold_value: The threshold value for binary inversion (default is 150).
    
    Returns:
    - mask: The resulting sketch effect image.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply a smaller Gaussian blur to retain more details
    gray_blur = cv2.GaussianBlur(gray, blur_kernel, 0)

    # Apply Canny edge detection with higher thresholds for detailed edges
    edges = cv2.Canny(gray_blur, canny_threshold1, canny_threshold2)

    # Combine with Laplacian for more edge detail
    laplacian = cv2.Laplacian(gray_blur, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))

    # Combine Canny and Laplacian edges for more detailed output
    combined_edges = cv2.add(edges, laplacian)

    # Threshold for binary inversion
    ret, mask = cv2.threshold(combined_edges, threshold_value, 255, cv2.THRESH_BINARY_INV)

    return mask


def main():
    # Initialize webcam capture
    capture = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not capture.isOpened():
        print("Error: Could not open video source.")
        return
    
    print("Press 'q' to quit the application.")
    
    # Loop to continuously capture frames from webcam
    while True:
        response, frame = capture.read()
        if not response:
            print("Failed to capture video frame.")
            break
        
        # Apply sketch effect
        sketched_frame = sketch(frame)

        # Show the result in a window
        cv2.imshow("Detailed Sketch Effect", sketched_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources and close windows
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
