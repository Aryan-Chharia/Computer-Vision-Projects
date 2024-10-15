import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical


# Update the path to where your CSV file is located
data = pd.read_csv('/content/fer2013.csv.zip')

# Display the first few rows to verify
print(data.head())

import numpy as np

# Preprocess images
def preprocess_images(data):
    images = []
    labels = []
    
    for index, row in data.iterrows():
        img = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8)
        img = img.reshape(48, 48)  # Reshape to 48x48
        images.append(img)
        labels.append(row['emotion'])

    images = np.array(images).reshape(-1, 48, 48, 1).astype('float32') / 255.0  # Normalize
    labels = np.array(labels)

    return images, labels

# Preprocess the images
images, labels = preprocess_images(data)

from sklearn.model_selection import train_test_split

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

print(f'Training data shape: {X_train.shape}, Training labels shape: {y_train.shape}')
print(f'Testing data shape: {X_test.shape}, Testing labels shape: {y_test.shape}')

# Load the dataset
data = pd.read_csv('/content/fer2013.csv.zip')

# Preprocess images
def preprocess_images(data):
    images = []
    labels = []
    
    for index, row in data.iterrows():
        img = np.fromstring(row['pixels'], sep=' ', dtype=np.uint8)
        img = img.reshape(48, 48)
        images.append(img)
        labels.append(row['emotion'])

    images = np.array(images).reshape(-1, 48, 48, 1).astype('float32') / 255.0
    labels = np.array(labels)

    return images, labels

# Preprocess the images
images, labels = preprocess_images(data)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# Create the CNN model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))  # 7 emotions

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Create the model
model = create_model()

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Visualize training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
