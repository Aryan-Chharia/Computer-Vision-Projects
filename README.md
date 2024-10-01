# Computer Vision Projects

Welcome to my Computer Vision Repository! This collection showcases advanced image processing techniques using Python and OpenCV. Whether you're a researcher, developer, or enthusiast, you'll find comprehensive insights and practical implementations to advance your computer vision skills.

## Projects

### 1. Image Stitching

The Image Stitching project demonstrates how to combine multiple images into a single panoramic image.

#### Features:

- Automatic detection and stitching of multiple images
- Handles different dataset names
- Post-processing to remove black borders
- Supports various OpenCV versions

#### Usage:

1. Place your images in the `Image Stitching Project/Dataset/[dataset_name]` folder.
2. Run `main.py` to stitch the images.
3. The results will be saved in the `Image Stitching Project/Result/[dataset_name]` folder.

### 2. Camera Calibration

This project focuses on camera calibration using a chessboard pattern, which is crucial for correcting lens distortion.

#### Features:

- Supports both regular and fisheye camera calibration
- Processes video input to find chessboard corners
- Generates camera matrix and distortion coefficients
- Includes an undistortion function to correct images

#### Usage:

1. Prepare a video of a chessboard pattern from various angles.
2. Adjust the input parameters in `calibration.py` as needed.
3. Run `calibration.py` to perform the calibration.
4. The camera parameters will be saved in a YAML file.
5. Use the undistort function to correct images using the calibrated parameters.

## Dependencies

- OpenCV
- NumPy
- imutils

## Installation

```bash
pip install opencv-python numpy imutils
```

## How to Run

1. Clone this repository:
   ```
   git clone https://github.com/your-username/Computer-Vision-Projects.git
   ```
2. Navigate to the project directory:
   ```
   cd Computer-Vision-Projects
   ```
3. Run the desired project:
   ```
   python main.py  # for Image Stitching
   python calibration.py  # for Camera Calibration
   ```

## Results

### Image Stitching

Original Image:

![Original Image](https://github.com/user-attachments/assets/a4ab67f7-3c74-4928-b8fb-e71787ea1c5e)

Stitched Result:

![Stitched Result](https://github.com/user-attachments/assets/f18e21cf-6be8-44d0-af7c-937161d5ff99)

### Camera Calibration

Video demonstration of the camera calibration process:

https://github.com/user-attachments/assets/30cad5b4-6a1a-49ea-839d-fc54f8a127c0

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
