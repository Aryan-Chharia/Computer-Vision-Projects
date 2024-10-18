# Facial Expression Recognition with CNN

## Overview

In this project, we leverage the power of Convolutional Neural Networks (CNNs) to recognize facial expressions using the **FER2013** dataset. The goal is to classify images into one of seven emotions: angry, disgust, fear, happy, sad, surprise, and neutral. This technology has a wide range of applications, from enhancing user experiences in social media to advancing emotional recognition systems in robotics.

## Table of Contents

- [Getting Started](#getting-started)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [License](#license)

## Getting Started

To get started with this project, you’ll need to set it up on your local machine. Follow the steps below:

### Dataset

This project uses the **FER2013 dataset**, which consists of grayscale images sized at 48x48 pixels. The dataset is labeled with seven different emotions, providing a robust foundation for training our model. You can download the dataset from [Kaggle](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data).

### Usage 

Download the Dataset: After downloading the FER2013 dataset, update the path in the code to point to your CSV file.

Run the code and execute python script "python main.py" 

### Model Architecture 

The CNN model architecture consists of several key components:

Convolutional Layers: Extract essential features from images.
Max Pooling Layers: Reduce the dimensionality of the feature maps, improving computational efficiency.
Dense Layers: Perform the final classification based on the learned features.
Dropout Layer: Prevent overfitting by randomly disabling neurons during training.

### Results 

Upon completion of the training process, the model's accuracy is visualized across epochs, offering valuable insights into its performance and the progression of its learning capabilities.

### License 

This project is licensed under the MIT License. For more details, please see the LICENSE file.

### Installation

Ensure you have Python installed on your machine. Set up a virtual environment and install the required packages using pip:

```bash
pip install pandas numpy matplotlib tensorflow scikit-learn 



