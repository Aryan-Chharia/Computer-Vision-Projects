# EfficientNet for Terrain Classification

## Overview
This project implements EfficientNet-B0 for terrain classification, achieving high accuracy in distinguishing between different terrain types.

### Model Performance
- **Validation Accuracy:** 91.93%
- **Test Accuracy:** 92.40%

## Model Architecture
EfficientNet is a convolutional neural network architecture that uses compound scaling to balance network depth, width, and resolution for optimal performance.

### Key Features
1. **Compound Scaling:** Uniformly scales depth/width/resolution using a compound coefficient.
2. **Mobile Inverted Bottleneck Convolution (MBConv):** Efficient building block that reduces parameters while maintaining performance.
3. **Squeeze-and-Excitation (SE) blocks:** Improves channel interdependencies at minimal computational cost.

### Architectural Diagram
```
Input Image
    │
    ▼
Stem Conv3x3
    │
    ▼
MBConv1, 3x3
    │
    ▼
MBConv6, 3x3
    │
    ▼
MBConv6, 5x5
    │
    ▼
MBConv6, 3x3
    │
    ▼
MBConv6, 5x5
    │
    ▼
MBConv6, 5x5
    │
    ▼
MBConv6, 3x3
    │
    ▼
Conv1x1 & Pooling
    │
    ▼
Fully Connected
    │
    ▼
Output (5 classes)
```
Each MBConv block includes Squeeze-and-Excitation optimization.

## Model Specifications (EfficientNet-B0)
| Metric              | Value              |
| ------------------- | ------------------ |
| Parameters          | ~4 million         |
| Model Size          | ~15.6 MB           |
| Memory Usage        | ~34.7 MB           |
| CPU Inference Speed | ~30-60ms per image |
| GPU Inference Speed | ~10-15ms per image |

*Note: Actual speed may vary based on hardware and image size.*

## Implementation Details

### Training Process
1. **Input Processing:** Images are resized to 224x224 and normalized.
2. **Feature Extraction:** EfficientNet backbone extracts hierarchical features.
3. **Global Pooling:** Features are pooled to create a fixed-size representation.
4. **Classification:** Fully connected layer maps features to 5 terrain classes.
5. **Output:** Softmax activation provides class probabilities.

### Key Training Parameters
- **Optimizer:** Adam
- **Learning Rate:** 1e-4
- **Batch Size:** 32
- **Loss Function:** Cross-entropy loss
- **Number of Epochs:** 25
- **Early Stopping:** Based on validation accuracy
- **Data Augmentation:** Horizontal flip, random crop, rotation

### Training and Optimization
We use transfer learning with a pre-trained EfficientNet-B0 model:
1. **Transfer Learning:** Pre-trained weights are loaded and initially frozen.
2. **Fine-tuning:** Top layers are unfrozen after the first few epochs.
3. **Batch Normalization:** Layers remain in training mode during fine-tuning.

### Evaluation Metrics
- Accuracy
- Confusion Matrix
- Precision, Recall, F1 Score
- Loss

## Results

### Video Demonstration
To see the model in action, check out our video demonstration:

https://github.com/user-attachments/assets/6c044349-035c-44d9-937f-9a6d079c5b08

## Usage

To use the model for terrain classification:
1. Update relevant paths in `transfer_learning.ipynb`
2. Run `transfer_learning.ipynb`

To load and use the pretrained model:
- Run `load_model.ipynb`

To check inference rates of the model:
- Run `cal_inference.ipynb`

For ways to improve model further please refer [Improvement.md](Improvement.md)
