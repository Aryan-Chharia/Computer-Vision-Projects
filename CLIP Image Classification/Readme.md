#  CLIP Image Classification

This project implements an  image classification system using OpenAI's CLIP (Contrastive Language-Image Pretraining) model. The system allows users to classify images based on dynamically generated textual prompts, leveraging state-of-the-art vision-language models to provide zero-shot classification capabilities. 

## Features

- **Asynchronous Image Processing**: Utilizes Python's `asyncio` to load and classify images concurrently, improving performance when handling large batches.
  
- **Dynamic Prompt Generation**: Automatically generates classification prompts based on user-defined base terms, allowing for flexible and contextual image queries.

- **Confidence Thresholding**: Filters classification results based on a user-defined confidence threshold, enhancing accuracy by omitting less certain predictions.

- **Multi-Modal Retrieval**: Enables users to retrieve images based on textual descriptions and vice versa, offering a versatile tool for various multi-modal tasks.

- **Robust Error Handling**: Includes comprehensive error handling and logging to help diagnose issues related to image loading and processing.

- **Batch Processing**: Supports processing multiple images from a specified folder, making it suitable for large datasets.

- **GPU Acceleration**: Automatically utilizes GPU for faster model inference if available, significantly improving processing times.

## Requirements

- Python 3.7 or higher
- PyTorch
- Transformers
- Pillow
- Matplotlib
- Asyncio

You can install the required packages using pip:

```bash
pip install torch torchvision torchaudio transformers pillow matplotlib
