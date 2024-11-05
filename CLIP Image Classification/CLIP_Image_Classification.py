import os
import asyncio
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load a pre-trained CLIP model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Cache to store results
cache = {}

async def load_image(image_path):
    """Asynchronously load an image."""
    try:
        img = Image.open(image_path)
        return img
    except Exception as e:
        logging.error(f"Error loading image {image_path}: {e}")
        return None

async def classify_image(image, prompts):
    """Classify a single image asynchronously."""
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().numpy()
    return probs

async def process_images(image_folder, base_terms, confidence_threshold=0.1):
    """Process all images in the folder asynchronously."""
    images = []
    prompts = [f"a {term}" for term in base_terms]

    # Load images asynchronously
    load_tasks = [load_image(os.path.join(image_folder, filename)) for filename in os.listdir(image_folder)]
    loaded_images = await asyncio.gather(*load_tasks)

    # Filter out None results
    loaded_images = [(img, filename) for img, filename in zip(loaded_images, os.listdir(image_folder)) if img is not None]

    # Classify images asynchronously
    classify_tasks = [classify_image(img, prompts) for img, _ in loaded_images]
    results = await asyncio.gather(*classify_tasks)

    # Filter results based on confidence threshold
    filtered_results = []
    for (img, filename), probs in zip(loaded_images, results):
        filtered_probs = [(prompt, prob) for prompt, prob in zip(prompts, probs[0]) if prob > confidence_threshold]
        filtered_results.append((filename, filtered_probs))

    return filtered_results

def display_results(results):
    """Display results with images."""
    for filename, probs in results:
        logging.info(f"\nResults for {filename}:")
        plt.imshow(Image.open(os.path.join(image_folder, filename)))
        plt.axis('off')
        plt.show()
        for prompt, prob in probs:
            print(f"Prompt: '{prompt}' - Probability: {prob:.4f}")

def main(image_folder, base_terms, confidence_threshold=0.1):
    """Main function to classify images in a folder."""
    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(process_images(image_folder, base_terms, confidence_threshold))
    display_results(results)

if __name__ == "__main__":
    # Example usage
    image_folder = "path_to_your_image_folder"  # Replace with your folder containing images
    base_terms = ["dog", "cat", "person", "scenic landscape", "cityscape at night", "delicious meal", "art painting", "futuristic robot"]
    confidence_threshold = 0.1  # Set confidence threshold for filtering results

    main(image_folder, base_terms, confidence_threshold)
