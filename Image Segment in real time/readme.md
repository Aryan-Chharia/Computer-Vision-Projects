# Image Segment in Real Time

This project demonstrates **real-time image segmentation** using advanced deep learning models. The project allows users to segment objects in static images. The segmentation model dynamically identifies and highlights various objects in the frame.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

---

## Project Overview

This project is built to achieve efficient **image segmentation** in real-time. Using a pre-trained segmentation model, such as FastSAM, users can segment objects from images and live streams. The system efficiently processes both static and dynamic image data, and the user can switch between the two modes seamlessly.

---

## Methodology

The project pipeline is built on the following steps:

1. **Input Image/Stream Handling**:
   - The system allows users to input static images or dynamic streams (such as from a webcam).
   - A function is added to **switch between image modes** dynamically, allowing flexibility in the type of input data.

2. **Preprocessing**:
   - Before the image is fed into the model, the frame is resized, normalized, and prepared for the segmentation model.

3. **Model Inference**:
   - The **FastSAM** model is used to segment the image. The model identifies and marks the boundaries of objects present in the frame.
   - It can work efficiently with both real-time video streams and static images.

4. **Postprocessing**:
   - The output from the model is processed to generate segmented areas on the image.
   - Segmentation masks are visualized by overlaying them on the original image in real-time.

5. **Real-Time Display**:
   - The segmented frames are displayed live, with each object in the frame highlighted by its segmentation mask.

---

## Results

Here are some results obtained using the system:

### Example 1: Static Image Segmentation

<img src="Image Segment in real time/input image.png" width="500"/>
<img src="Image Segment in real time/output.png" width="500"/> 

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jaidh01/Computer-Vision-Projects.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Image%20Segment%20in%20real%20time
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. To run segmentation on static images:
   ```bash
   python segment_image.py --image path_to_image
   ```

2. To run real-time segmentation on a video stream (webcam):
   ```bash
   python segment_realtime.py
   ```

---

## Contributing

Feel free to contribute to this project by submitting a pull request. Any enhancements, bug fixes, or suggestions are welcome.
