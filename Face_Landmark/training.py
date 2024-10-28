import time
import cv2
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imutils
import matplotlib.image as mpimg
from math import radians, cos, sin
import xml.etree.ElementTree as ET
import urllib.request
import tarfile
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

# Set device to GPU if available


# Dataset download and extraction
def download_and_extract_dataset():
    dataset_url = "http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz"
    dataset_tar = "ibug_300W_large_face_landmark_dataset.tar.gz"
    extracted_folder = "ibug_300W_large_face_landmark_dataset"

    if not os.path.exists(extracted_folder):
        print("Downloading the dataset...")
        urllib.request.urlretrieve(dataset_url, dataset_tar)
        print("Download completed.")

        print("Extracting the dataset...")
        with tarfile.open(dataset_tar, "r:gz") as tar:
            tar.extractall()
        print("Extraction completed.")

        print("Cleaning up...")
        os.remove(dataset_tar)
        print("Cleanup completed.")
    else:
        print(f"{extracted_folder} already exists. Skipping download.")

# download_and_extract_dataset()

# Data visualization function
def visualize_landmarks(image, landmarks, title=None):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=5, c='g')
    if title:
        plt.title(title)
    plt.show()

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

        image = imutils.rotate(np.array(image), angle)
        landmarks = landmarks - 0.5
        new_landmarks = np.matmul(landmarks, transformation_matrix)
        new_landmarks = new_landmarks + 0.5
        return Image.fromarray(image), new_landmarks

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

class FaceLandmarksDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.image_filenames, self.landmarks, self.crops = self.parse_xml()

    def parse_xml(self):
        tree = ET.parse(os.path.join(self.root_dir, 'labels_ibug_300W_train.xml'))
        root = tree.getroot()

        image_filenames, landmarks, crops = [], [], []
        for filename in root[2]:
            image_filenames.append(os.path.join(self.root_dir, filename.attrib['file']))
            crops.append(filename[0].attrib)
            landmark = [[int(filename[0][num].attrib['x']), int(filename[0][num].attrib['y'])] for num in range(68)]
            landmarks.append(landmark)

        return image_filenames, np.array(landmarks).astype('float32'), crops

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):
        image = cv2.imread(self.image_filenames[index], 0)
        landmarks = self.landmarks[index]
        
        if self.transforms:
            image, landmarks = self.transforms(image, landmarks, self.crops[index])

        landmarks = landmarks - 0.5
        return image, landmarks

class ResNetFaceLandmarks(nn.Module):
    def __init__(self, num_classes=136):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10):
    model.to(device)
    best_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        
        for images, landmarks in train_loader:
            images, landmarks = images.to(device), landmarks.view(landmarks.size(0), -1).to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for images, landmarks in valid_loader:
                images, landmarks = images.to(device), landmarks.view(landmarks.size(0), -1).to(device)
                outputs = model(images)
                loss = criterion(outputs, landmarks)
                valid_loss += loss.item()
        
        valid_loss /= len(valid_loader)
        
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'best_face_landmarks_model.pth')
            print("Model saved!")
        
        print('-' * 40)
    
    print("Training complete!")

def visualize_predictions(model, valid_loader, num_samples=8):
    model.eval()
    model.to(device)
    
    images, landmarks = next(iter(valid_loader))
    images, landmarks = images.to(device), landmarks.to(device)
    
    with torch.no_grad():
        predictions = model(images)
    
    images = images.cpu().numpy()
    landmarks = (landmarks.cpu() + 0.5) * 224
    predictions = (predictions.cpu().view(-1, 68, 2) + 0.5) * 224
    
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 5 * num_samples))
    
    for i in range(num_samples):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].scatter(predictions[i, :, 0], predictions[i, :, 1], c='r', s=5, label='Predicted')
        axes[i].scatter(landmarks[i, :, 0], landmarks[i, :, 1], c='g', s=5, label='Ground Truth')
        axes[i].legend()
        axes[i].set_title(f"Sample {i+1}")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    dataset = FaceLandmarksDataset('ibug_300W_large_face_landmark_dataset', Transforms())
    train_size = int(0.9 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=4)

    model = ResNetFaceLandmarks()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs=10)

    best_model = ResNetFaceLandmarks()
    best_model.load_state_dict(torch.load('best_face_landmarks_model.pth'))
    visualize_predictions(best_model, valid_loader)
