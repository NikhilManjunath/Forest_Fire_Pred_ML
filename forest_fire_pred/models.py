"""
Model Training and Selection
"""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.exceptions import FitFailedWarning, ConvergenceWarning
from typing import Any, Dict, Tuple, List, Union
import forest_fire_pred.utils as utils
import warnings
import pandas as pd

# Suppress Fit Fail Warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", ConvergenceWarning)
    warnings.simplefilter("ignore", FitFailedWarning)

class ModelTrainer:
    """
    Handles model training and cross-validation for different classifiers.
    """
    def __init__(self, model_name: str):
        """
        Initialize ModelTrainer with a model ('SVM', 'KNN', 'Ridge', 'Logistic_Regression', 'Random_Forest').
        """
        self.model_name = model_name

        # Models to Try
        self.models: Dict[str, Dict[str, Any]] = {
            'SVM': {
                'model': SVC(),
                'param_grid': {
                    'C': [0.1, 1, 10, 100, 1000],
                    'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                    'kernel': ['rbf', 'linear', 'poly']
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'param_grid': {
                    'n_neighbors': list(range(1, 11)),
                    'weights': ['uniform', 'distance'],
                    'metric': ['minkowski', 'chebyshev', 'euclidean']
                }
            },
            'Ridge': {
                'model': RidgeClassifier(max_iter=5000),
                'param_grid': {
                    'alpha': [0.1, 1, 10],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
                }
            },
            'Logistic_Regression': {
                'model': LogisticRegression(),
                'param_grid': {
                    'solver': ['newton-cg', 'liblinear', 'lbfgs'],
                    'penalty': ['l2']
                }
            },
            'Random_Forest': {
                'model': RandomForestClassifier(),
                'param_grid': {
                    'max_features': ['sqrt', 'log2'],
                    'criterion': ['gini', 'entropy'],
                    'n_estimators': [100, 200, 300, 350]
                }
            },
        }

    def train_and_cross_validation(self, X_train: Any, y_train: Any) -> Tuple[Any, float, Dict[str, Any]]:
        """
        Train the model using cross-validation and grid search.
        Args:
            X_train: Training features.
            y_train: Training labels.
        Returns:
            Tuple of (model instance, best F1 score, best hyperparameters).
        """
        if self.model_name not in self.models:
            raise ValueError(f'Model {self.model_name} is not supported')
        
        print(f'Running training and cross validation for: {self.model_name}')
        model_info = self.models[self.model_name]
        model = model_info['model']
        param_grid = model_info['param_grid']

        # CrossValidated Hyperparameter Search
        try:
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='f1', cv=5)
            grid_search.fit(X_train, y_train)
            best_params, best_score = grid_search.best_params_, grid_search.best_score_

        except Exception as e:
            print(f"Grid search failed for {self.model_name}: {e}")
            raise

        return model, best_score, best_params

class Experiments:
    """
    Runs experiments for model and feature selection.
    """
    def __init__(self):
        """
        Initialize Experiments with a list of models and placeholders for best results.
        """
        self.model_list: List[str] = ['SVM', 'KNN', 'Ridge', 'Logistic_Regression', 'Random_Forest']
        self.best_model: Any = None
        self.best_score: float = 0.0
        self.best_params: Union[None, Dict[str, Any]] = None
        self.best_data: str = ''

    def data_experiments(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[str, Any, Dict[str, Any], float, Any]:
        """
        Run experiments for combinations of feature selection techniques and model
        Args:
            X_train: Training features.
            y_train: Training labels.
        Returns:
            Tuple of (feature selection type, best model, best params, best score, feature selector).
        """

        # Feature Selector 1: PCA
        PCA, X_train_PCA = utils.PCA_func(X_train, 9)

        # Feature Selector 2: SFS
        sfs_columns = utils.seq_feat_selector(X_train, y_train, 9)
        X_train_SFS = X_train[sfs_columns]

        # Dict of feature transformed Train Data
        data_dict = {
            'PCA': X_train_PCA,
            'SFS': X_train_SFS
        }

        # Run Experiments
        results = []
        for data in data_dict:
            print(f'\nTraining on {data} features')
            for model in self.model_list:
                trainer = ModelTrainer(model)

                try:
                    model_fn, score, params = trainer.train_and_cross_validation(data_dict[data], y_train)

                except Exception as e:
                    print(f"Training failed for {model} on {data}: {e}")
                    continue

                results.append({
                    'Feature_Set': data,
                    'Model': model,
                    'F1_Score': round(score, 2),
                    'Best_Params': params
                })

                # Track best model, score and parameters from experiments
                if score > self.best_score:
                    self.best_data, self.best_model, self.best_score, self.best_params = data, model_fn, score, params

        # Display Experiments results as a table, sorted by decreasing order of cross validation scores
        results_df = pd.DataFrame(results)
        print("=== Experiment Results ===")
        print("\n" + results_df.sort_values(by='F1_Score', ascending=False).to_string(index=False))

        # Return best feature selector and best model combo
        if self.best_data == 'PCA':
            return self.best_data, self.best_model, self.best_params, self.best_score, PCA
        else:
            return self.best_data, self.best_model, self.best_params, self.best_score, sfs_columns