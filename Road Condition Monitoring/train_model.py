# train_model.py

import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

# Import RoadConditionDataset from notebook
from road_condition_monitor import RoadConditionDataset  # assuming dataset code in notebook is modularized here

# Data transformation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Model
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
# Similar code as the notebook
