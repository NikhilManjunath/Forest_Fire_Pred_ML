import argparse
from forest_fire_pred import Dataloader
from forest_fire_pred import DataPreprocessor
from forest_fire_pred import plot_confusion_matrix
from forest_fire_pred import compute_metrics
import joblib
import os
import logging
from typing import Tuple, Any
import pandas as pd

def load_models(path: str) -> Tuple[Any, Any, dict, dict]:
    """
    Load trained model (and other saved processing transforms) from the specified directory.
    """
    try:
        # Load Trained Polynomial Feature Transform Function
        poly = joblib.load(os.path.join(path, 'poly.pkl'))
        # Load Feature Scaler
        scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        # Load Feature Selector
        feature_selection = joblib.load(os.path.join(path, 'feature_selection.pkl'))
        # Load Inference Model
        inference_model = joblib.load(os.path.join(path, 'best_model.pkl'))

        return poly, scaler, feature_selection, inference_model
    
    except FileNotFoundError as e:
        print(f"Model artifact not found: {e}")
        raise

    except Exception as e:
        print(f"Failed to load model artifacts: {e}")
        raise

def parse_single_input(input_str: str) -> pd.DataFrame:
    """
    Parse a comma-separated string into a DataFrame for single prediction.
    Args:
        input_str: Comma-separated string of values in order: date,temp,rh,ws,rain,ffmc,dmc,dc,isi,bui,class
                  Format: "date,temp,rh,ws,rain,ffmc,dmc,dc,isi,bui,class"
                  Example: "01-09-2012,15.86139179,103.0834932,23.92998243,6.862310716,18.20680273,6.705574999,16.93405947,1.272175111,1.567136112,0"
    Returns:
        DataFrame with a single row for prediction.
    """

    feature_names = [
        'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI'
    ]

    values = input_str.split(',')
    if len(values) != 11:  # date + 9 features + class
        raise ValueError(f"Expected 11 values (date + 9 features + class), got {len(values)}.")
    
    # Parse values in order: Date,Temperature,RH,Ws,Rain,FFMC,DMC,DC,ISI,BUI,Classes
    date = values[0]  # Date string
    feature_values = [float(x) for x in values[1:10]]  # Features (positions 1-9)
    class_value = int(values[10])  # Class value (position 10)
    
    # Create DataFrame with features, date, and class
    df = pd.DataFrame([dict(zip(feature_names, feature_values))])
    df['Date'] = date
    df['Classes'] = class_value
    
    return df

def main() -> None:
    """
    Main entry point for testing the Forest Fire Prediction ML model.
    """
    parser = argparse.ArgumentParser(description="Forest Fire Prediction ML Pipeline")
    parser.add_argument('--test-data', type=str, help='Path to test data CSV (required if not using --predict-single)')
    parser.add_argument('--model-dir', type=str, required=True, help='Directory to load saved model and preprocessing artifacts')
    parser.add_argument('--predict-single', type=str, help='Comma-separated feature values for single prediction')
    args = parser.parse_args()

    # Validate arguments
    if not args.predict_single and not args.test_data:
        print("Must provide either --predict-single or --test-data.")
        return

    # Load artifacts
    try:
        poly, scaler, feature_selector, model = load_models(args.model_dir)

    except Exception as e:
        print(f"Failed to load model artifacts: {e}")
        return

    # Preprocessing
    preprocessor = DataPreprocessor(poly, scaler)

    # If single input prediction
    if args.predict_single:
        try:
            single_df = parse_single_input(args.predict_single)
            X_single = single_df.copy()

            # Feature Selection
            if feature_selector['type'] == 'PCA':
                X_single_processed = preprocessor.transform(X_single)
                X_single_fs = feature_selector['features'].transform(X_single_processed)
            else:
                X_single_processed = preprocessor.transform(X_single)
                X_single_fs = X_single_processed[feature_selector['features']]

            # Predict
            best_model, hyperparams = model['model'], model['hyperparameters']
            best_model.set_params(**hyperparams)
            prediction = 'No Fire' if best_model.predict(X_single_fs)[0] == 0 else 'Fire'
            actual = 'No Fire' if X_single['Classes'].iloc[0] == 0 else 'Fire'
            print(f'Prediction: {prediction}')
            print(f'Actual: {actual}')

        except Exception as e:
            logging.error(f"Failed to predict single input: {e}")
            print(f"Error: {e}")
        return

    # If not single prediction, run test set evaluation
    try:
        # Loading Test data
        test_loader = Dataloader(args.test_data)
        test = test_loader.load_data()
    except Exception as e:
        logging.error(f"Failed to load test data: {e}")
        return

    # Preprocessing for test set
    X_test = preprocessor.transform(test)
    y_test = test['Classes']

    # Features
    if feature_selector['type'] == 'PCA':
        X_test_fs = feature_selector['features'].transform(X_test)
    else:
        X_test_fs = X_test[feature_selector['features']]

    # Testing on Test Data
    print('Testing Models on Test Data')
    best_model, hyperparams = model['model'], model['hyperparameters']
    best_model.set_params(**hyperparams)
    y_test_pred = best_model.predict(X_test_fs)
    metrics = compute_metrics(y_test_pred, y_test.to_numpy())
    print(f'Performance of Best ML Model on Test Data: {metrics}')
    plot_confusion_matrix(y_test, y_test_pred)

if __name__ == "__main__":
    main()

