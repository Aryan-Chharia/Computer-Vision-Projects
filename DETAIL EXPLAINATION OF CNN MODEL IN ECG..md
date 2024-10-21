CHANGES IN README.md FOR ECG MODEL
DETAILED EXPLAINATION OF CNN MODEL ARCHITECTURE

Input Layer: This layer remains the same, taking the preprocessed ECG image as input.

Convolutional Layers: We now have three convolutional layers, each with an increasing number of filters to progressively capture more complex spatial features from the ECG images. Each convolutional layer uses the ReLU (Rectified Linear Unit) activation function to introduce non-linearity.

The first layer extracts basic features like edges and corners.
The second layer identifies more complex patterns.
The third layer captures high-level representations of the image.
Pooling Layers: Each convolutional layer is followed by a pooling layer, specifically max-pooling, which reduces the dimensionality of the feature maps while retaining important information. This not only reduces computational cost but also helps prevent overfitting.

Flatten Layer: After the final pooling layer, the data is flattened into a one-dimensional vector, allowing it to be passed into fully connected layers.

Fully Connected Layers: These layers combine the extracted features to learn high-level representations. There’s one fully connected layer with a set number of neurons (for example, 128), using the ReLU activation function.

Output Layer: The final layer uses a softmax activation function and contains four neurons, corresponding to the four ECG categories, to predict the probability distribution across those categories.
