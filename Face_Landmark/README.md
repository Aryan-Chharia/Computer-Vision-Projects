# Facial Landmark Detection Project

This project implements a facial landmark detection system using PyTorch and a ResNet18 architecture. It trains on the iBUG 300-W dataset to predict 68 facial landmarks and includes functionality to use the trained model on new images.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Usage](#usage)
6. [Model Architecture](#model-architecture)
7. [Training](#training)
8. [Prediction on New Images](#prediction-on-new-images)
9. [Visualization](#visualization)
10. [License](#license)

## Prerequisites

- Python 3.7+
- PyTorch
- torchvision
- OpenCV (cv2)
- NumPy
- Matplotlib
- Pillow

## Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/facial-landmark-detection.git
   cd facial-landmark-detection
   ```

2. Install the required packages:
   ```
   pip install torch torchvision opencv-python numpy matplotlib pillow
   ```

## Dataset

This project uses the iBUG 300-W dataset. The training script includes a function to download and extract the dataset automatically. If you haven't downloaded it yet, uncomment the following line in the main training script:

```python
 download_and_extract_dataset()
```

## Project Structure

- `main.py`: The main script containing all the code for training the model.
- `predict.py`: Script for using the trained model to predict landmarks on new images.
- `best_face_landmarks_model.pth`: The best model weights saved during training.
- `README.md`: This file.

## Usage

### Training the Model

Run the main script:

```
python main.py
```

This will train the model and visualize predictions on the validation set.

### Predicting on New Images

After training the model, you can use it to predict facial landmarks on new images:

```
python predict.py
```

Make sure to update the `test_image_path` in the `main()` function of `predict.py` to point to your test image.

## Model Architecture

The model uses a modified ResNet18 architecture:

- The input layer is changed to accept grayscale images (1 channel).
- The final fully connected layer is modified to output 136 values (68 landmarks \* 2 coordinates).

## Training

The model is trained for 10 epochs using the Adam optimizer and Mean Squared Error (MSE) loss. The training process includes:

- Data augmentation (rotation, color jitter)
- Training/validation split (90/10)
- Best model saving based on validation loss

## Prediction on New Images

The `predict.py` script provides functionality to use the trained model on new images. It includes the following steps:

1. Loading the trained model
2. Preprocessing the input image
3. Predicting facial landmarks
4. Visualizing the results

To use this script:

1. Ensure you have a trained model file (`best_face_landmarks_model.pth`).
2. Update the `test_image_path` in the `main()` function to point to your test image.
3. Run the script: `python predict.py`

The script will display the input image with the predicted facial landmarks overlaid.

## Visualization

Both the training script and the prediction script include visualization functions to display the facial landmarks on the images.
