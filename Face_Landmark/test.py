import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
import torch.nn as nn
from PIL import Image
import random
import warnings
from math import radians, cos, sin

# Define the ResNetFaceLandmarks model
class ResNetFaceLandmarks(nn.Module):
    def __init__(self, num_classes=136):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

# Define the Transforms class
class Transforms:
    def __init__(self):
        self.color_jitter = transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
        )

    def rotate(self, image, landmarks, angle):
        angle = random.uniform(-angle, +angle)
        transformation_matrix = torch.tensor([
            [cos(radians(angle)), -sin(radians(angle))], 
            [sin(radians(angle)), cos(radians(angle))]
        ])

        image = np.array(image)
        image = Image.fromarray(np.uint8(image))
        image = image.rotate(angle)
        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return image, new_landmarks

    def resize(self, image, landmarks, img_size):
        image = transforms.Resize(img_size)(image)
        return image, landmarks

    def crop_face(self, image, landmarks, crops):
        left, top = int(crops['left']), int(crops['top'])
        width, height = int(crops['width']), int(crops['height'])

        image = transforms.functional.crop(image, top, left, height, width)
        img_shape = np.array(image).shape
        landmarks = torch.tensor(landmarks) - torch.tensor([[left, top]])
        landmarks = landmarks / torch.tensor([img_shape[1], img_shape[0]])
        return image, landmarks

    def __call__(self, image, landmarks, crops):
        image = Image.fromarray(image)
        image, landmarks = self.crop_face(image, landmarks, crops)
        image, landmarks = self.resize(image, landmarks, (224, 224))
        image, landmarks = self.color_jitter(image), landmarks
        image, landmarks = self.rotate(image, landmarks, angle=10)
        
        image = transforms.ToTensor()(image)
        image = transforms.Normalize([0.5], [0.5])(image)
        return image, landmarks

def load_model(model_path):
    model = ResNetFaceLandmarks()
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path):
    image = cv2.imread(image_path, 0)
    transform = Transforms()
    image_tensor, _ = transform(image, np.zeros((68, 2)), {'left': 0, 'top': 0, 'width': image.shape[1], 'height': image.shape[0]})
    return image_tensor.unsqueeze(0)

def predict_landmarks(model, image_tensor):
    with torch.no_grad():
        predictions = model(image_tensor)
    landmarks = (predictions.view(-1, 68, 2) + 0.5) * 224
    return landmarks.squeeze().numpy()

def visualize_landmarks(image_path, landmarks):
    image = cv2.imread(image_path, 0)
    image = cv2.resize(image, (224, 224))
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=5, c='r')
    plt.title("Predicted Facial Landmarks")
    plt.show()

def main():
    warnings.filterwarnings("ignore")
    model_path = 'best_face_landmarks_model.pth'
    test_image_path = r'path/to/elkfdgldgimage'  # Replace with your test image path

    # Load the trained model
    model = load_model(model_path)

    # Preprocess the test image
    image_tensor = preprocess_image(test_image_path)

    # Predict landmarks
    predicted_landmarks = predict_landmarks(model, image_tensor)

    # Visualize the results
    visualize_landmarks(test_image_path, predicted_landmarks)

if __name__ == "__main__":
    main()