
# CAPTCHA OCR with VisionEncoderDecoder Model

This project is an Optical Character Recognition (OCR) system built using the VisionEncoderDecoder Model (specifically, Microsoft's TrOCR model) to recognize and decode text from CAPTCHA images. The model was trained and evaluated on a dataset of CAPTCHA images, aiming to predict text accurately from challenging and varied CAPTCHA examples.

## Project Overview

The project consists of several stages:

1. **Data Setup**: The project fetches and prepares CAPTCHA images from Kaggle for training, validation, and testing.
2. **Model Training**: The VisionEncoderDecoder model is trained on CAPTCHA images, with a custom loss function designed for character error rate (CER) evaluation.
3. **Model Evaluation**: After each epoch, the model is evaluated using CER on the validation dataset to monitor accuracy and performance.
4. **Prediction Interface**: A GUI was created to test the model on randomly selected test CAPTCHA images, displaying the predicted text.

## Installation

To use this project, ensure you have access to Google Colab. Then, you can run the notebook with the following steps:

1. Install required dependencies:

    ```bash
    !pip install kaggle
    !pip install -q transformers evaluate jiwer
    ```

2. Download the dataset and extract:

    ```bash
    !kaggle datasets download -d topkek69/captcha
    !unzip captcha -d extracted_data
    ```

3. Run the notebook and follow the steps.

## Data Preparation

The dataset is divided into three directories for training, validation, and testing. Each image corresponds to a CAPTCHA image file, and the file name (excluding the `.png` extension) is used as the label.

- **Training Set**: Used to train the model.
- **Validation Set**: Used to evaluate the model's accuracy after each epoch.
- **Test Set**: Used to test the final model performance.

## Model Architecture

This project uses Microsoft's **TrOCR (Text Recognition OCR)** model, a VisionEncoderDecoder model designed for OCR tasks. Key parameters include:

- **Optimizer**: AdamW with a learning rate of `5e-5`.
- **Evaluation Metric**: Character Error Rate (CER).
- **Beam Search Parameters**: `num_beams = 2`, `max_length = 10`.

## GUI for CAPTCHA Text Generation

A GUI interface using `ipywidgets` allows users to generate text predictions for random CAPTCHA images from the test set. The generated text is displayed upon button click.

## Model Saving and Loading

The trained model is saved locally and can be loaded again using the following code:

```python
from transformers import VisionEncoderDecoderConfig, VisionEncoderDecoderModel

encoder_decoder_config = VisionEncoderDecoderConfig.from_pretrained("OCR-model")
OCR_model = VisionEncoderDecoderModel.from_pretrained("OCR-model", config=encoder_decoder_config).to(device)
```

## Conclusion

This project demonstrates how to apply a VisionEncoderDecoder model for OCR tasks, specifically for CAPTCHA decoding. The model shows promising results, although real-world CAPTCHA complexity may require further tuning and dataset expansion.


