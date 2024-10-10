# 🐶 End-to-End Multi-class Dog Breed Classification

This project builds an end-to-end multi-class image classifier using TensorFlow 2.x and TensorFlow Hub to identify the breed of a dog given its image.

## 📖 Table of Contents
- [Problem](#problem)
- [Data](#data)
- [Evaluation](#evaluation)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Training](#model-training)
- [Results](#results)

## Problem

Given an image of a dog, the objective is to identify its breed. The model should be able to classify the image into one of the 120 dog breeds available in the dataset.

## Data

The dataset used is from the [Kaggle Dog Breed Identification competition](https://www.kaggle.com/c/dog-breed-identification/data). It contains:
- **Training set**: Over 10,000 labeled images of 120 different dog breeds.
- **Test set**: Over 10,000 unlabeled images for prediction.

## Evaluation

The evaluation metric is based on a file containing prediction probabilities for each breed of each test image, following the [Kaggle evaluation protocol](https://www.kaggle.com/competitions/dog-breed-identification/overview/evaluation).

## Features

- Multi-class classification with 120 dog breeds.
- Deep learning using TensorFlow and transfer learning techniques.
- The model processes and predicts probabilities for each test image.

## Requirements

To install the necessary packages, you can use the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/dog-breed-classification.git
   cd dog-breed-classification
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**: Open the notebook and execute the cells step by step to train the model and make predictions.

## Model Training

The model uses transfer learning with pre-trained models from TensorFlow Hub. The notebook includes steps to preprocess data, set up the model architecture, train the model, and evaluate its performance.

## Results

The results are saved in the notebook as prediction probabilities for each dog breed for the test set images. The notebook demonstrates the accuracy and evaluation metrics.
