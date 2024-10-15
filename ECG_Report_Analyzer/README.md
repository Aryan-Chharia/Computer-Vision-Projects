## ECG Report Classification with Convolutional Neural Networks

This project implements a Convolutional Neural Network (CNN) model for classifying ECG reports into four categories:

* Abnormal Heartbeat
* History of Myocardial Infarction (MI)
* Myocardial Infarction
* Normal

### Dataset

The project uses the ECG dataset available on Mendeley Data [ECG Images dataset of Cardiac Patients](https://data.mendeley.com/). This dataset contains preprocessed ECG signals labeled according to the following scheme:

* *label_map = {'Abnormal_Heartbeat': 0, 'MI_history': 1, 'Myocardial_Infarction': 2, 'Normal':3}*

### Model Architecture

The project employs a CNN architecture for image classification. The specific details of the architecture, such as the number of layers, activation functions, and hyperparameters, are likely to be defined in your code. This README provides a general outline.

**Typical CNN Architecture:**

1. **Input Layer:** Receives the preprocessed ECG signal as an image.
2. **Convolutional Layers:** Extract spatial features from the ECG image through multiple convolutional operations. These layers can use activation functions like ReLU to introduce non-linearity.
3. **Pooling Layers:** Downsize the feature maps while retaining important information through techniques like max pooling.
4. **Flatten Layer:** Convert the output of the convolutional layers into a one-dimensional vector suitable for feeding into fully-connected layers.
5. **Fully-Connected Layers:** Combine the extracted features to learn higher-level representations.
6. **Output Layer:** Contains a single neuron with a softmax activation function to predict the probability distribution of the four ECG categories.

### Training and Evaluation

The project trains the CNN model on a portion of the dataset. The remaining data is used for validation and testing to evaluate the model's performance. Common metrics used for classification tasks include:

* Accuracy: Proportion of correctly classified samples.
* **Model has accuracy of 95%.**
