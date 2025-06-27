# Forest Fire Prediction using Machine Learning

This repository contains a machine learning pipeline for predicting forest fire occurrences based on meteorological data.

## Project Overview

This project implements a complete ML pipeline for forest fire prediction, including:
- **Data Loading & Preprocessing**: Automated data loading and feature engineering
- **Model Training & Selection**: Multiple ML algorithms with hyperparameter tuning
- **Feature Selection**: PCA and Sequential Feature Selection methods
- **Model Evaluation**: Comprehensive metrics and visualization
- **Single Prediction**: Real-time prediction on individual data points

## Project Structure

```
Forest_Fire_Pred_ML/
├── forest_fire_pred/           # Main package
│   ├── __init__.py            # Package initializer
│   ├── data/                  # Dataset directory
│   │   ├── algerian_fires_train.csv
│   │   └── algerian_fires_test.csv
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessing.py       # Data preprocessing pipeline
│   ├── models.py              # Model training and selection
│   ├── metrics.py             # Evaluation metrics
│   ├── utils.py               # Helper functions
│   ├── visualization.py       # Plotting functions
│   └── saved_models/          # Trained model artifacts
├── tests/                     # Unit tests
├── main.py                    # Training pipeline entry point
├── test.py                    # Testing and prediction entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### Prerequisites
- Python 3.9+
- pip package manager

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/NikhilManjunath/Forest_Fire_Pred_ML.git
   cd Forest_Fire_Pred_ML
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The project uses the Algerian Forest Fires Dataset from the UCI Repository (https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset)

## Usage

### Training the Model

To train and save the best training model:

```bash
python3 train.py --train-data forest_fire_pred/data/algerian_fires_train.csv \
                --save-dir forest_fire_pred/saved_models
```

**Arguments:**
- `--train-data`: Path to training data CSV (required)
- `--save-dir`: Directory to save model artifacts (optional)

### Testing on Full Dataset

To evaluate the model on the test set:

```bash
python3 test.py --test-data forest_fire_pred/data/algerian_fires_test.csv \
                --model-dir forest_fire_pred/saved_models
```

### Single Prediction

To predict on a single data point:

```bash
python3 test.py --model-dir forest_fire_pred/saved_models \
                --predict-single "01-06-2012,25.0,45.0,10.0,0.0,85.0,20.0,200.0,5.0,30.0,0"
```

**Single Prediction Format:**
```
Date,Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,Classes
```

**Example:**
- Date: `01-06-2012` (DD-MM-YYYY format)
- Temperature: `25.0` (°C)
- RH: `45.0` (Relative Humidity %)
- Ws: `10.0` (Wind Speed km/h)
- Rain: `0.0` (mm)
- FFMC: `85.0` (Fine Fuel Moisture Code)
- DMC: `20.0` (Duff Moisture Code)
- DC: `200.0` (Drought Code)
- ISI: `5.0` (Initial Spread Index)
- BUI: `30.0` (Buildup Index)
- Classes: `0` (0: No Fire, 1: Fire)

## Testing

Run the test suite:

```bash
python3 -m pytest tests/
```