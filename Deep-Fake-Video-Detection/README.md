# DeepFake Video Detection and Time Series Prediction using LSTM

## Overview

This project consists of two main tasks:
1. **DeepFake Video Detection**: Using a pre-trained model based on InceptionV3 and GRU layers, we aim to classify videos as either "REAL" or "FAKE".
2. **LSTM Sequence Prediction**: A simple Long Short-Term Memory (LSTM) model predicts the next value in a sequence based on prior values using a sine wave dataset.

---

## Table of Contents

- [Requirements](#requirements)
- [DeepFake Video Detection](#deepfake-video-detection)
  - [Dataset](#dataset)
  - [Feature Extraction](#feature-extraction)
  - [Modeling](#modeling)
  - [Training and Evaluation](#training-and-evaluation)
  - [Video Prediction](#video-prediction)
- [LSTM Time Series Prediction](#lstm-time-series-prediction)
  - [Dataset](#dataset-1)
  - [Model Architecture](#model-architecture)
  - [Training and Evaluation](#training-and-evaluation-1)

---

## Requirements

Install the required packages:

```bash
pip install opencv-contrib-python imageio tensorflow matplotlib pandas numpy
```

---

## DeepFake Video Detection

### Dataset
- Download the dataset from the following link: https://drive.google.com/drive/folders/1BWEzxq0exLO_8vinYsQERCHlGBChQ5i8?usp=sharing
- The training and test datasets consist of DeepFake videos, with metadata for labeling as "REAL" or "FAKE".
- Dataset paths:
  - `TRAIN_SAMPLE_FOLDER`: `C:\Users\Spandana\Gen AI Internship\Deep_fake\dataset\train_sample_videos`
  - `TEST_FOLDER`: `C:\Users\Spandana\Gen AI Internship\Deep_fake\dataset\test_videos`
  
### Feature Extraction

We use the **InceptionV3** pre-trained model to extract meaningful features from video frames. Each video is sampled frame-by-frame, and features are extracted and processed.

### Modeling

The model is built using **Gated Recurrent Units (GRU)** to classify the video based on sequential frame features. The architecture consists of:
- Two GRU layers
- Dropout for regularization
- Dense layers for classification

### Training and Evaluation

- The model is trained using binary cross-entropy loss and Adam optimizer.
- Performance is measured using accuracy metrics, and the best model weights are saved.

```python
history = model.fit(
    [train_data[0], train_data[1]],
    train_labels,
    validation_data=([test_data[0], test_data[1]], test_labels),
    callbacks=[checkpoint],
    epochs=70,
    batch_size=8
)
```

### Video Prediction

Once the model is trained, you can predict whether a given video is **REAL** or **FAKE**:

```python
if(sequence_prediction(test_video) >= 0.5):
    print(f'The predicted class of the video is REAL')
else:
    print(f'The predicted class of the video is FAKE')
```

---

## LSTM Time Series Prediction

### Dataset

- A sine wave sequence is generated for time series prediction:
  
```python
sequence = create_sequence(1000)
```

- The dataset is structured for the LSTM model with a look-back period, i.e., the number of previous steps used to predict the next value.

### Model Architecture

- The model consists of two stacked **LSTM** layers followed by a dense output layer.
  
```python
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
model.add(LSTM(50))
model.add(Dense(1))
```

### Training and Evaluation

- The model is trained using the mean squared error (MSE) loss function and the Adam optimizer.
  
```python
model.fit(X, y, epochs=20, batch_size=32)
```

- After training, the model's predictions are plotted against the true sequence for comparison:

```python
plt.plot(sequence[look_back:], label="True Sequence")
plt.plot(predictions, label="Predicted Sequence")
plt.legend()
plt.show()
```
