# README

## Overview

This project involves building a deep learning model to classify medical images as either **normal** or **abnormal** (potentially indicating cancer) using a convolutional neural network (CNN). The dataset includes two categories: **normal** and **abnormal** images, and the model is trained on images resized to 150x150 pixels.

### Key Features:
- **Data Loading**: The script loads images from `train` and `test` directories and resizes them for training and testing.
- **Model**: A CNN model using TensorFlow/Keras to classify images.
- **Visualization**: Includes visualizations such as dataset class distribution and example images.
- **Prediction**: The model predicts whether an image is **normal** or **abnormal**, with suggestions based on the predictions.

## Setup

### Prerequisites:
Ensure you have the following libraries installed:
- `numpy`
- `os`
- `sklearn`
- `seaborn`
- `matplotlib`
- `opencv-python` (cv2)
- `tensorflow`
- `tqdm`
- `pandas`

You can install them using pip:
```bash
pip install numpy scikit-learn seaborn matplotlib opencv-python tensorflow tqdm pandas
```

### Folder Structure:
The project expects the following directory structure for loading images:
```
project/
│
├── train/
│   ├── normal/
│   ├── abnormal/
│
├── test/
│   ├── normal/
│   ├── abnormal/
```
Each folder contains images related to the respective class.

### Image Requirements:
- Images should be in standard formats (e.g., PNG, JPEG).
- Images in the **train** and **test** directories should belong to either the `normal` or `abnormal` folders.

## Code Explanation

### Loading Data:
The `load_data()` function reads the training and testing data from the respective directories, converts them into arrays, and resizes each image to **150x150 pixels** for model input.

```python
def load_data():
    datasets = [r'train', r'test']
    ...
    return output
```

### Model Architecture:
The model is a simple CNN with the following layers:
- **2 Conv2D Layers**: Extract features using 3x3 filters.
- **2 MaxPooling Layers**: Downsample the feature maps.
- **Flatten Layer**: Converts 2D data to 1D.
- **Dense Layers**: Fully connected layers for classification.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    ...
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

### Model Compilation and Training:
The model is compiled using the Adam optimizer and trained for 30 epochs. The loss function used is `sparse_categorical_crossentropy`, which is suitable for multi-class classification.

```python
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=128, epochs=30, validation_split = 0.2)
```

### Visualization:
- **Class Distribution**: Displays a bar chart showing the distribution of **normal** and **abnormal** images in the dataset.
- **Example Images**: Displays a 5x5 grid of randomly selected images from the training dataset.

```python
pd.DataFrame({'train': train_counts, 'test': test_counts}, index=class_names).plot.bar()
display_examples(class_names, train_images, train_labels)
```

### Model Evaluation:
The test set is evaluated using the model:
```python
test_loss = model.evaluate(test_images, test_labels)
```

### Predictions:
The model predicts whether a new image is **normal** or **abnormal**, displaying appropriate suggestions based on the prediction.

```python
test_image = image.load_img(r"test/abnormal/000002 (4).png", target_size = (150, 150))
predictions = model.predict(test_image)
pred_labels = np.argmax(predictions, axis = 1)
suggestions1()
```

The suggestions are:
- **NORMAL**: "Everything looks fine, go for regular checkup."
- **ABNORMAL**: "Seems to have cancer, consult the doctor immediately."

### Pie Chart:
A pie chart is generated to visualize the distribution of **normal** and **abnormal** images in the training set.

```python
plt.pie(train_counts, explode=(0,0,), labels=class_names)
```

## How to Run

1. **Prepare the Dataset**: Organize your dataset into `train` and `test` directories, each containing `normal` and `abnormal` subdirectories.
2. **Run the Script**: Execute the script to load the dataset, train the model, and make predictions on test images.
3. **View Results**: After training, you can use the model to predict new images from the test set.

## Suggestions and Next Steps

- Consider tuning the model by adding more convolutional layers or adjusting hyperparameters like the learning rate and number of epochs.
- You may want to try data augmentation to increase the robustness of your model, especially if the dataset is small.
- Add more complex evaluation metrics such as precision, recall, and F1-score to better understand model performance. 

