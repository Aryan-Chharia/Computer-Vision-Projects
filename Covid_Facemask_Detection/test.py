import torch
from torchvision import transforms, models
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import warnings
# Define class labels
class_map = {0: "with_mask", 1: "without_mask", 2: "mask_weared_incorrect"}

# Define image transformations (same as used during training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def load_model(model_path='face_mask_model.pth'):
    """Load the trained model from the given path."""
    model = models.resnet34(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)  # 3 output classes
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode
    return model

def predict_image(image_path, model):
    """Predict the class of the given image."""
    image = Image.open(image_path).convert("RGB")
    transformed_image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(transformed_image)
        _, predicted = torch.max(output, 1)
    
    class_label = class_map[predicted.item()]
    return class_label, image

def show_image_with_prediction(image, prediction):
    """Display the image with the predicted label."""
    plt.imshow(image)
    plt.title(f"Prediction: {prediction}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description="Predict mask status for an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--model_path", type=str, default="face_mask_model.pth", help="Path to the trained model")
    args = parser.parse_args()

    # Load the model and generate prediction
    model = load_model(args.model_path)
    prediction, image = predict_image(args.image_path, model)

    print(f"Prediction: {prediction}")
    show_image_with_prediction(image, prediction)
