# ECG Autoencoder

**Anomaly detection through reconstruction error · Python / TensorFlow**

An exploratory notebook using the ECG5000 dataset. A dense autoencoder learns to reconstruct normal heartbeats; reconstruction errors are then compared across normal and abnormal classes.

## Approach

1. Separate the five ECG classes and fit a MinMax scaler on the normal training signals.
2. Train a `140 → 32 → 16 → 8 → 16 → 32 → 140` autoencoder with Adam and mean absolute error.
3. Plot reconstructed signals and compare error distributions.
4. Explore a threshold defined as the mean normal reconstruction error plus one standard deviation.

## Files

| File | Purpose |
| --- | --- |
| [AUTOENCODERS ECG.ipynb](AUTOENCODERS%20ECG.ipynb) | Data preparation, model training and visual analysis |
| `ECG5000_train.csv` | Training data |
| `ECG5000_test.csv` | Test data used in the notebook |

## Explore

Open the notebook from the repository root in Jupyter. It imports NumPy, pandas, TensorFlow, scikit-learn and Matplotlib. Training is configured for 20 epochs.

## Scope

This is a learning experiment on ECG reconstruction and anomaly detection. Despite the repository name, the notebook does not train on noisy/clean signal pairs. The test data also contributes to validation and threshold selection, so this is not an independent final evaluation or a clinically validated detector.
