# Traffic_Sign_Detection

Using German Traffic Sign Recognition Benchmark, in this script I have used
Data Augmentation and then passed the images to miniVGGNet (VGG 7) a famous architecture of
Convolutional Neural Network (CNN).

It contains 138 Million parameters and is a simplistic application of VGG-16 and VGG-19. In the end
a model with an accuracy of 99.7 percent has been obtained.

## Setup

ALl the requirements can be found in [`requirements.txt`](requirements.txt).

simply use `pip install -r requirements.txt`. You can also install `tensorflow-gpu` depending upon the
tensorflow version of yours, use any search engine for more information as Tensorflow-GPU
is a complex and large topic in itself.

## Model

The [Jupyter Notebook](trafffic_sign_detection.ipynb) contains all the relevant information about the Model.

## Database

I have used German Traffic Sign Recognition Benchmark Dataset or _GTSRB_, it can be downloaded from
[here](https://benchmark.ini.rub.de/).

## Usage

The [Jupyter Notebook](trafffic_sign_detection.ipynb) should be referred for the Usage guidelines, while the Python
[script](traffic_sign_detection.py) for a better understanding of documentation.



