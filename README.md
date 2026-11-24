# EEMD Water Leak Acoustic Detection

A Python project for detecting water leaks from acoustic signals using Ensemble Empirical Mode Decomposition (EEMD), handcrafted signal features, and a Random Forest classifier.

The project supports two workflows:

1. Training a classifier using experimental water-distribution-system recordings.
2. Running inference on simulated sensor windows.

## Features

- Simulates pump noise and realistic leak acoustic signatures.
- Decomposes each signal into intrinsic mode functions using EEMD.
- Extracts energy, kurtosis, RMS, standard deviation, crest factor, zero-crossing rate, and spectral centroid features.
- Uses LASSO for feature selection.
- Addresses class imbalance with ADASYN.
- Trains a Random Forest leak detector.
- Provides a simulated real-time prediction loop.
- Saves and reloads the trained model with `joblib`.

## Project Structure

```text
EEMD_project/
├── config.py           # Sampling, EEMD, model, and dataset settings
├── pre_process.py      # EEMD decomposition and feature extraction
├── simulate_data.py    # Synthetic pump and leak signal generation
├── train.py            # Dataset loading, feature selection, and model training
├── predict.py          # Model loading and simulated inference
├── data/
│   └── simulated.csv   # Generated synthetic dataset
├── models/
│   └── leak_detector.pkl
└── .vscode/
    └── settings.json
```

## Installation

Python 3.9 or later is recommended.

Clone the repository:

```bash
git clone https://github.com/YassmineAalla/EEMD_project.git
cd EEMD_project
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install numpy pandas scipy PyEMD scikit-learn imbalanced-learn joblib
```

## Configuration

The main settings are in `config.py`.

```python
FS = 30780
WINDOW_SEC = 0.5
N_SAMPLES = 15390
RF_TREES = 200
LEAK_THRESHOLD = 0.70
DATA_PATH = "data/simulated.csv"
MODEL_PATH = "models/leak_detector.pkl"
```

Before training with the experimental dataset, update:

```python
DATASET_PATH = r"path\to\the\dataset"
SENSOR = "A1"
```

A1 refers to the first accelerometer channel. The training code searches for CSV or RAW sensor files and determines labels from filenames:

- `LC`, `CC`, `GL`, or `OL`: leak
- `NL`: no leak

## Generating a Synthetic Dataset

Create a simulated dataset containing pump noise, random turbulence, and occasional leak bursts:

```bash
python simulate_data.py
```

This creates `data/simulated.csv`. The default configuration generates 1,000 windows, with approximately 15% containing leaks.

## Training the Model

Training requires the experimental dataset and the path configured in `config.py`.

```bash
python train.py
```

The training pipeline:

1. Loads sensor recordings.
2. Splits each recording into 0.5-second windows.
3. Applies EEMD to every window.
4. Extracts 42 features from six intrinsic mode functions.
5. Selects informative features using LASSO.
6. Uses ADASYN to balance the training classes.
7. Trains a balanced Random Forest.
8. Reports precision, recall, F1-score, and support.
9. Saves the model and selected feature indices to `models/leak_detector.pkl`.

## Running Prediction

Run inference using generated pump windows with simulated leak injections:

```bash
python predict.py
```

Every seventh window receives a synthetic leak signal. The console prints:

- A leak or clear status
- Leak probability
- Processing time
- Whether the window contained an injected leak

## Replacing Simulated Input

The function `read_window_simulated()` in `predict.py` is the intended integration point for real sensor acquisition. Replace it with ADC or file-stream reading code that returns exactly `N_SAMPLES` samples:

```python
def read_window_simulated(inject_leak=False):
    # Read N_SAMPLES samples from the sensor
    return sensor_samples
```

## EEMD and Extracted Features

For every 15,390-sample window:

- EEMD produces six intrinsic mode functions.
- Each IMF contributes seven features:

  - Energy
  - Kurtosis
  - RMS
  - Standard deviation
  - Crest factor
  - Zero-crossing rate
  - Spectral centroid

The resulting feature vector has 42 values.

## Known Limitations

- `config.py` contains a machine-specific Windows dataset path.
- The repository commits a local Python virtual environment and generated cache files.
- `predict.py` currently uses simulated data instead of a physical ADC.
- The EEMD implementation may be computationally expensive for real-time use.
- The synthetic dataset is intended for demonstration and is not a substitute for validation on experimental recordings.

## License

No license is currently specified.
