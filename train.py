import argparse
from forest_fire_pred.data_loader import Dataloader
from forest_fire_pred.preprocessing import DataPreprocessor
from forest_fire_pred.models import Experiments
from forest_fire_pred.visualization import plot_confusion_matrix
from forest_fire_pred.metrics import compute_metrics
import joblib
import os
import logging

def main() -> None:
    """
    Main entry point for training ML Models on the Forest Fire dataset.
    """
    parser = argparse.ArgumentParser(description="Forest Fire Training ML Pipeline")
    parser.add_argument('--train-data', type=str, required=True, help='Path to training data CSV')
    # Optional save directory: only save if this is explicitly set
    parser.add_argument('--save-dir', type=str, help='Directory to save model and preprocessing artifacts (optional)')
    args = parser.parse_args()

    try:
        # Loading Train data
        train_loader = Dataloader(args.train_data)
        train = train_loader.load_data()
    except Exception as e:
        logging.error(f"Failed to load data: {e}")
        return

    # Preprocessing
    preprocessor = DataPreprocessor()
    X_train, y_train = preprocessor.fit(train)

    # Model Selection and Hyperparameter Tuning
    exp = Experiments()
    fs_type, model, params, score, features = exp.data_experiments(X_train, y_train)
    print(f"\nBest Training Model: {model}({params}), F1 Score: {score}")

    # Train best model on entire train data
    if fs_type == 'PCA':
        X_train_fs = features.transform(X_train)
    else:
        X_train_fs = X_train[features]
    model.set_params(**params)
    model.fit(X_train_fs, y_train)

    # Save Model
    if args.save_dir:
        try:
            print(f"\nSaving model to: {args.save_dir}")
            joblib.dump(preprocessor.poly, os.path.join(args.save_dir, "poly.pkl"))
            joblib.dump(preprocessor.scaler, os.path.join(args.save_dir, "scaler.pkl"))
            joblib.dump({
                'type': fs_type,
                'features': features
            }, os.path.join(args.save_dir, "feature_selection.pkl"))
            joblib.dump({
                'model': model,
                'hyperparameters': params
            }, os.path.join(args.save_dir, "best_model.pkl"))

        except Exception as e:
            print(f"Failed to save model artifacts: {e}")
    else:
        print("No save directory provided. Skipping model saving.")

if __name__ == "__main__":
    main()











    

        