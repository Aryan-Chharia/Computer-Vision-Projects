# Face Mask Detection Script

This script uses a pre-trained PyTorch model to detect whether a person in an image is wearing a face mask, wearing it incorrectly, or not wearing a mask at all.

## Prerequisites

- Python 3.6+
- PyTorch
- torchvision
- Pillow
- matplotlib
- argparse

You can install the required packages using pip:

```
pip install torch torchvision Pillow matplotlib argparse
```

## Usage

1. Ensure you have the trained model file `face_mask_model.pth` in the same directory as the script.

2. Run the script from the command line, providing the path to the image you want to analyze:

```
python face_mask_detection.py path/to/your/image.jpg
```

3. Optionally, you can specify a different path for the model:

```
python face_mask_detection.py path/to/your/image.jpg --model_path path/to/your/model.pth
```

## How it works

1. The script loads a pre-trained ResNet34 model that has been fine-tuned for face mask detection.
2. It processes the input image to match the format used during training.
3. The processed image is passed through the model to get a prediction.
4. The prediction is mapped to one of three classes: "with_mask", "without_mask", or "mask_weared_incorrect".
5. The script displays the original image with the prediction as the title.

## Output

The script will print the prediction to the console and display the image with the prediction as the title.

## Note

This script assumes that the input image contains a face. It does not perform face detection, so for best results, use images that are already cropped to show a single face.

## Troubleshooting

If you encounter any issues related to CUDA or GPU, try running the script on CPU by modifying the `map_location` parameter in the `load_model` function:

```python
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
```

## License

This project is open-source and available under the MIT License.
