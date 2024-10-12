import cv2     # for capturing videos
import math   # for mathematical operations
import matplotlib.pyplot as plt    # for plotting the images
%matplotlib inline
import pandas as pd
from keras.preprocessing import image   # for preprocessing the images
import numpy as np    # for mathematical operations
from keras.utils import np_utils
from skimage.transform import resize   # for resizing images

class DataGenerator(keras.utils.Sequence):
#Generates data for Keras
def init(self, list_IDs, labels, batch_size=32, dim=(32,32), n_channels=1, n_classes=10, shuffle=True):

#Initialization
self.dim = dim
self.batch_size = batch_size
self.labels = labels
self.list_IDs = list_IDs
self.n_channels = n_channels
self.n_classes = n_classes
self.shuffle = shuffle
self.on_epoch_end()

def __len__(self):
    'Denotes the number of batches per epoch'
    return int(np.floor(len(self.list_IDs) / self.batch_size))

def __getitem__(self, index):
    'Generate one batch of data'
    # Generate indexes of the batch
    indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

    # Find list of IDs
    list_IDs_temp = [self.list_IDs[k] for k in indexes]

    # Generate data
    X, y = self.__data_generation(indexes)

    return X, y

def on_epoch_end(self):
    'Updates indexes after each epoch'
    self.indexes = np.arange(len(self.list_IDs))
    if self.shuffle == True:
        np.random.shuffle(self.indexes)
        
def read_and_resize(self, filepath):
    img = imread('/home/eren/dataset_scrap/' + filepath)
    res = resize(img, (150, 150), preserve_range=True, mode='reflect')
    return np.expand_dims(res, 0)
        
def __data_generation(self, list_IDs_temp):
    'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
    # Initialization
    X = np.empty((self.batch_size, *self.dim, self.n_channels))
    y = np.empty((self.batch_size), dtype=int)
    
    X = [self.read_and_resize(self.list_IDs[i])
         for i in range(0, len(list_IDs_temp))]
    y = self.labels[:len(list_IDs_temp)]
    X = np.vstack(X)
    
    return X, y`


#frame sampling
#sampling mode 1 
#sampling mode 2
#get_chunks
#sample aug
#save frame sampling 
