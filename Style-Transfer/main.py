import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load an image
def load_and_process_image(image_path):
    img = Image.open(image_path)
    img = img.resize((256, 256))  # Resize to the input size expected by the model
    img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1] and convert to float32
    return np.expand_dims(img, axis=0)

# Deprocess the image
def deprocess_image(image):
    img = image.squeeze()
    img = np.clip(img, 0, 1)  # Ensure values are between 0 and 1
    img = (img * 255).astype(np.uint8)  # Convert back to uint8
    return Image.fromarray(img)

# Load the pre-trained model from TensorFlow Hub
model_url = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
stylization_model = hub.load(model_url)

# Style transfer function
def style_transfer(content_image_path, style_image_path):
    content_image = load_and_process_image(content_image_path)
    style_image = load_and_process_image(style_image_path)

    # Run the model
    stylized_image = stylization_model(tf.constant(content_image), tf.constant(style_image))[0]

    return deprocess_image(stylized_image.numpy())

# Example usage
content_path = 'c1.jpg'  # Replace with your content image path
style_path = 's1.jpg'      # Replace with your style image path
result_image = style_transfer(content_path, style_path)

# Show the result
plt.imshow(result_image)
plt.axis('off')
plt.show()

