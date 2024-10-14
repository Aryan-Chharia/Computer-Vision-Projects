import tensorflow as tf
from tensorflow.keras.layers import Dense,Convolution2D, Input,Flatten,Dropout,MaxPooling2D, Conv2D
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder

classifier = Sequential()

classifier.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(128, 128, 1)))
#classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
#classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.25))

classifier.add(Flatten())
classifier.add(Dense(512, activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(27, activation='softmax'))

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory("Sign2/train",
                                                 target_size=(128, 128),
                                                 batch_size=10,
                                                 color_mode='grayscale',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory("Sign2/test",
                                            target_size=(128 , 128),
                                            batch_size=10,
                                            color_mode='grayscale',
                                            class_mode='categorical') 

classifier.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history=classifier.fit_generator(
        training_set,
        steps_per_epoch=1285,
        epochs=5,
        validation_data=test_set,
        validation_steps=427)

model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
print('Model Saved')
classifier.save_weights('model-bw.h5')
print('Weights saved')
