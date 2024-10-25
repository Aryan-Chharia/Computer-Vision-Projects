import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2

# Define the model architecture (same as in the training script)
class DeblurCNN(nn.Module):
    def __init__(self):
        super(DeblurCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=10, stride=1, padding='same')
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 320, kernel_size=1, stride=1, padding='same')
        self.bn2 = nn.BatchNorm2d(320)
        self.conv3 = nn.Conv2d(320, 320, kernel_size=1, stride=1, padding='same')
        self.bn3 = nn.BatchNorm2d(320)
        self.conv4 = nn.Conv2d(320, 320, kernel_size=1, stride=1, padding='same')
        self.bn4 = nn.BatchNorm2d(320)
        self.conv5 = nn.Conv2d(320, 128, kernel_size=1, stride=1, padding='same')
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same')
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding='same')
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512, 128, kernel_size=5, stride=1, padding='same')
        self.bn8 = nn.BatchNorm2d(128)
        self.conv9 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding='same')
        self.bn9 = nn.BatchNorm2d(128)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same')
        self.bn10 = nn.BatchNorm2d(128)
        self.conv11 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding='same')
        self.bn11 = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=5, stride=1, padding='same')
        self.bn12 = nn.BatchNorm2d(128)
        self.conv13 = nn.Conv2d(128, 256, kernel_size=1, stride=1, padding='same')
        self.bn13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256, 64, kernel_size=7, stride=1, padding='same')
        self.bn14 = nn.BatchNorm2d(64)
        self.conv15 = nn.Conv2d(64, 3, kernel_size=7, stride=1, padding='same')

    def forward(self, x):
        x = nn.ReLU()(self.bn1(self.conv1(x)))
        x = nn.ReLU()(self.bn2(self.conv2(x)))
        x = nn.ReLU()(self.bn3(self.conv3(x)))
        x = nn.ReLU()(self.bn4(self.conv4(x)))
        x = nn.ReLU()(self.bn5(self.conv5(x)))
        x = nn.ReLU()(self.bn6(self.conv6(x)))
        x = nn.ReLU()(self.bn7(self.conv7(x)))
        x = nn.ReLU()(self.bn8(self.conv8(x)))
        x = nn.ReLU()(self.bn9(self.conv9(x)))
        x = nn.ReLU()(self.bn10(self.conv10(x)))
        x = nn.ReLU()(self.bn11(self.conv11(x)))
        x = nn.ReLU()(self.bn12(self.conv12(x)))
        x = nn.ReLU()(self.bn13(self.conv13(x)))
        x = nn.ReLU()(self.bn14(self.conv14(x)))
        x = nn.ReLU()(self.conv15(x))
        return x

# Function to process a single image
def process_image(path):
    img = cv2.imread(path)
    img = np.asarray(img, dtype="float32")
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # Change to channel-first format
    return img

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeblurCNN().to(device)
model.load_state_dict(torch.load('deblur_cnn_model.pth', map_location=device))
model.eval()

# Function to deblur an image
def deblur_image(image_path):
    # Process the image
    input_image = process_image(image_path)
    input_tensor = torch.from_numpy(input_image).unsqueeze(0).to(device)
    
    # Deblur the image
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    # Convert the output tensor to a numpy array
    output_image = output_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    # Clip the values to be between 0 and 1
    output_image = np.clip(output_image, 0, 1)
    
    return input_image.transpose(1, 2, 0), output_image

# Test the model on a single image
test_image_path = r"C:\Users\agraw\Downloads\images.jpeg"  # Replace with the path to your test image
input_image, deblurred_image = deblur_image(test_image_path)

# Display the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.imshow(input_image)
ax1.set_title('Input (Blurred)')
ax2.imshow(deblurred_image)
ax2.set_title('Output (Deblurred)')
plt.savefig('deblur_test_result.png')
plt.show()

print("Deblurring complete. Results saved as 'deblur_test_result.png'.")