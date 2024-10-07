# LSTM Model for Sequence Prediction

This project demonstrates a simple LSTM (Long Short-Term Memory) model that predicts the next value in a sine wave sequence based on previous values. The model is evaluated using Mean Squared Error (MSE).

## Requirements

To run the project, you need the following libraries:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- Scikit-learn

You can install the required libraries using `pip`:

```bash
pip install tensorflow numpy matplotlib scikit-learn
```

## Dataset
The dataset is generated using a sine wave function. The sequence contains 1000 data points generated from a sine wave.

## Model Architecture
The model consists of the following layers:

LSTM Layer (50 units, return_sequences=True): The first LSTM layer that returns sequences.
LSTM Layer (50 units): The second LSTM layer.
Dense Layer (1 unit): A dense layer to output the next value in the sequence.
Training
The model is trained on the sine wave sequence data using the Mean Squared Error loss function and the Adam optimizer. The look-back period (how many previous values to consider) is set to 5.

## Evaluation
After training the model, it is evaluated using Mean Squared Error (MSE). You can see the MSE value printed in the output.

## Usage
To run the code and evaluate the model, use the following steps:

Generate a sine wave sequence.
Prepare the dataset with a look-back period of 5.
Build and train the LSTM model.
Evaluate the model using Mean Squared Error (MSE).
Visualize the true sequence and predicted sequence using Matplotlib.
