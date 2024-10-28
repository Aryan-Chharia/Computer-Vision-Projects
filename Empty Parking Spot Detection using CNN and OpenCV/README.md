
# Empty Parking Spot Detection using CNN and OpenCV

## Abstract

This project develops an automated system to detect empty parking slots using Convolutional Neural Networks (CNNs) and OpenCV. The CNN model, trained on labeled parking slot images, classifies slots as empty or occupied based on grayscale inputs, aiming to optimize parking resource utilization and reduce urban traffic congestion.

## Objectives

- Develop an automated parking slot detection system using CNNs and OpenCV.
- Enhance detection accuracy through image preprocessing techniques.
- Provide real-time analysis of parking slot occupancy.

## Technologies Used

- Keras
- OpenCV
- Python
- CNN (Convolutional Neural Network)

## Methodology

1. **Image Preprocessing**: Convert images to grayscale and apply filters to enhance features.
2. **Model Training**: Train a CNN on labeled parking slot images to classify empty or occupied slots.
3. **Detection**: Use the trained model to identify and mark empty parking slots in real-time images.

## Results

- **Training Accuracy**: 94.24%
- **Validation Accuracy**: 97.11%

This project contributes to smart city applications by improving parking management efficiency and urban mobility.

## How to Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Deekshu966/Mini-Project
   cd empty-parking-spot-detection
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Code**:

   - **To run 1.ImageProcessing**:
     ```bash
     python 1.ImageProcessing.py
     ```

   - **To run 2.parkingspotcoordinate.py**:
     ```bash
     python 2.parkingspotcoordinate.py
     ```

   - **To run 3.emptyparkingspotdetectionmodel.py**:
     ```bash
     python 3.emptyparkingspotdetectionmodel.py
     ```

   - **To run 4.emptyparkingspotdetection.py**:
     ```bash
     python 4.emptyparkingspotdetection.py
     ```

## Contribution

Feel free to fork this repository, submit issues, and send pull requests.

## License

This project is licensed under the MIT License.

