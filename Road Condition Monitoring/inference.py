# inference.py

import torch
import cv2
from torchvision import transforms

# Load model and transform
model = torch.load('models/model_checkpoint.pth')  # Load saved model
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def predict(image_path):
    image = cv2.imread(image_path)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    output = model(image)
    _, predicted = torch.max(output, 1)
    return 'Smooth' if predicted.item() == 0 else 'Pothole/Crack'

if __name__ == "__main__":
    test_image = "path_to_test_image.jpg"
    print(predict(test_image))
