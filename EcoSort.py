import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Directory structure: /data/{class_name}/image.jpg
data_dir = './data/'

# Image parameters
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32

# Data augmentation to improve model generalization
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2,  # 80% train, 20% validation
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()
EPOCHS = 10

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Save the model for future use
model.save('waste_classification_model.h5')
import cv2

# Load the trained model
model = tf.keras.models.load_model('waste_classification_model.h5')

# Load an image for prediction
image_path = 'test_image.jpg'
img = cv2.imread(image_path)
img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
img = np.expand_dims(img, axis=0) / 255.0  # Rescale as in training

# Perform prediction
pred = model.predict(img)
pred_class = np.argmax(pred, axis=1)

# Map the prediction index back to class label
class_labels = {v: k for k, v in train_generator.class_indices.items()}
print(f"Predicted class: {class_labels[pred_class[0]]}")
# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the converted model
with open('waste_classification_model.tflite', 'wb') as f:
    f.write(tflite_model)

