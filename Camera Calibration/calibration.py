import os
import numpy as np
import cv2

# Input parameters defined in a dictionary
input_params = {
    "input": "Camera Calibration/video.mp4",  # Path to input video file
    "grid": "10x7",  # Size of the calibration grid pattern
    "resolution": "640x640",  # Resolution of the camera image
    "framestep": 5,  # Use every nth frame in the video
    "output": "Camera Calibration\yaml\params.yaml",  # Path to output YAML file
    "fisheye": True,  # Set true if this is a fisheye camera
    "flip": 0,  # Flip method of the camera
    "no_gst": False  # Set true if not using GStreamer for camera capture
}

# Directory to save the camera parameters file
TARGET_DIR = os.path.join(os.getcwd(), "Camera Calibration\yaml")
DEFAULT_PARAM_FILE = os.path.join(TARGET_DIR, "camera_params.yaml")

def main():
    # Parse resolution and grid size
    W, H = map(int, input_params["resolution"].split("x"))
    grid_size = tuple(map(int, input_params["grid"].split("x")))

    # Prepare object points
    grid_points = np.zeros((1, np.prod(grid_size), 3), np.float32)
    grid_points[0, :, :2] = np.indices(grid_size).T.reshape(-1, 2)

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    # Create directory if it doesn't exist
    if not os.path.exists(TARGET_DIR):
        os.mkdir(TARGET_DIR)

    text1 = "Press 'c' to calibrate"
    text2 = "Press 'q' to quit"
    text3 = f"Device: {input_params['input']}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.6

    quit = False
    do_calib = False
    i = -1

    # Open video capture
    vdo = cv2.VideoCapture(input_params["input"])

    while True:
        i += 1
        success, img = vdo.read()
        if not success:
            continue

        if i % input_params["framestep"] != 0:
            continue

        print(f"Searching for chessboard corners in frame {i}...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, 
            grid_size, 
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
        )

        if found:
            term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
            print("OK")
            imgpoints.append(corners)
            objpoints.append(grid_points)
            cv2.drawChessboardCorners(img, grid_size, corners, found)

        cv2.putText(img, text1, (20, 70), font, fontscale, (255, 200, 0), 2)
        cv2.putText(img, text2, (20, 110), font, fontscale, (255, 200, 0), 2)
        cv2.putText(img, text3, (20, 30), font, fontscale, (255, 200, 0), 2)
        cv2.imshow("Corners", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            print("\nPerforming calibration...\n")
            if len(objpoints) < 12:
                print(f"Less than 12 corners ({len(objpoints)}) detected, calibration failed")
                continue
            else:
                do_calib = True
                break
        elif key == ord("q"):
            quit = True
            break

    if quit:
        cv2.destroyAllWindows()

    if do_calib:
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        calibration_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
            cv2.fisheye.CALIB_CHECK_COND +
            cv2.fisheye.CALIB_FIX_SKEW
        )

        if input_params["fisheye"]:
            ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objpoints, 
                imgpoints, 
                (W, H), 
                K, 
                D, 
                None, 
                None, 
                calibration_flags, 
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
            )
        else:
            ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, 
                imgpoints, 
                (W, H), 
                None, 
                None
            )

        if ret:
            fs = cv2.FileStorage(input_params["output"], cv2.FILE_STORAGE_WRITE)
            fs.write("resolution", np.int32([W, H]))
            fs.write("camera_matrix", K)
            fs.write("dist_coeffs", D)
            fs.release()
            print("Successfully saved camera data")
            cv2.putText(img, "Success!", (220, 240), font, 2, (0, 0, 255), 2)
        else:
            cv2.putText(img, "Failed!", (220, 240), font, 2, (0, 0, 255), 2)

        cv2.imshow("Corners", img)
        cv2.waitKey(0)

def undistort():
    # Load camera parameters
    fs = cv2.FileStorage(input_params["output"], cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("camera_matrix").mat()
    dist_coeffs = fs.getNode("dist_coeffs").mat()
    resolution = tuple(map(int, fs.getNode("resolution").mat().reshape(2).tolist()))

    # Open the video file
    vdo = cv2.VideoCapture(input_params["input"])
    if not vdo.isOpened():
        raise ValueError("Error opening video file")

    scale = 1
    shift = 0

    while True:
        success, img = vdo.read()
        if not success:
            print("No more frames to read or error reading the video file.")
            break

        # Adjust camera matrix
        scaled_camera_matrix = camera_matrix.copy()
        scaled_camera_matrix[[0, 1], [0, 1]] *= scale
        scaled_camera_matrix[:2, 2] += shift

        # Initialize undistort rectify map
        x, y = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix, 
            dist_coeffs, 
            None, 
            scaled_camera_matrix, 
            resolution, 
            cv2.CV_32F
        )

        # Apply the remap function
        undistorted_image = cv2.remap(img, x, y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

        # Display the frame
        cv2.imshow('Undistorted Image', undistorted_image)
        cv2.imshow('Image', img)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    vdo.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    undistort()
