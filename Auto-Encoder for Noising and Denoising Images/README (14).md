Description:

DenoiseNet is a powerful image processing project that utilizes a convolutional autoencoder to effectively remove noise from images, enhancing their quality and visual appeal. This project focuses on training a deep learning model to reconstruct clean images from their noisy counterparts, demonstrating the practical applications of neural networks in computer vision.

Key Components:
Dataset Utilization:

The project uses the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes. This dataset is ideal for image denoising tasks due to its diverse range of images.
Noising Mechanism:

To simulate real-world scenarios, noise is artificially added to the images using a Gaussian noise model. This helps in training the autoencoder to learn the characteristics of both noisy and clean images.
Model Architecture:

The core of DenoiseNet is built upon a convolutional autoencoder architecture. The model consists of:
Encoder: Multiple convolutional layers followed by max pooling layers to extract features from the noisy images, reducing the spatial dimensions while retaining essential information.
Decoder: Up-sampling and convolutional layers that reconstruct the original image from the compressed representation learned by the encoder.
Training Process:

The model is trained using the Mean Squared Error (MSE) loss function, which quantifies the difference between the predicted (denoised) images and the actual clean images. The Adam optimizer is employed to enhance the training efficiency.
The model undergoes 50 epochs of training, allowing it to learn the intricacies of image denoising effectively.
Results Visualization:

The project visualizes the results by displaying the original noisy images, the denoised images produced by the autoencoder, and the corresponding clean images. This comparison showcases the effectiveness of the autoencoder in noise reduction.
Applications:

DenoiseNet has significant applications in fields like photography, video processing, medical imaging, and any area where image quality is paramount. It can be used to enhance images captured in low-light conditions or those affected by various types of noise. Please note During preprocessing, CIFAR-10 images are often normalized, augmented, or scaled in such a way that fine details are lost. For instance:
Normalization: Normalizing the pixel values to a range (e.g., between 0 and 1) may make images appear a bit less sharp when visualized.
Augmentation: Random cropping or resizing can also reduce image clarity.(For better clarity one can use images from ImageNet or CIFAR 100) 
Sample model output:
![Sample output ](https://github.com/user-attachments/assets/b137edae-89a2-43a8-b157-15128137e92f)

