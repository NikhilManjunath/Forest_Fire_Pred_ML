"""
Data Preprocessing Pipeline
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from typing import Tuple, Optional
import logging

class DataPreprocessor:
    """
    Preprocessing pipeline that handles feature engineering, polynomial expansion, and feature scaling.
    """
    def __init__(self, poly: Optional[PolynomialFeatures] = None, scaler: Optional[StandardScaler] = None):
        """
        Initialize the DataPreprocessor.
        Args:
            poly: Optional pre-fitted PolynomialFeatures transformer
            scaler: Optional pre-fitted StandardScaler.
        """
        self.poly = poly
        self.scaler = scaler
        self.feature_columns = [
            'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI'
        ]

    def fit(self, train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        This function is run during training to fit the polynomial feature expansion and feature scaler on the training data.
        Args:
            train: Training DataFrame with required columns.
        Returns:
            Tuple of (X_train_transformed, y_train)
        """

        train = train.copy()
        # Add Date Features
        train = self._add_date_features(train)

        # Split features from labels
        X = train[self.feature_columns]
        y = train['Classes']

        # Polynomial Feature Expansion
        self.poly = PolynomialFeatures(2)
        X_expanded = self.poly.fit_transform(X)
        X_expanded = pd.DataFrame(X_expanded, columns=self.poly.get_feature_names_out())
        if '1' in X_expanded.columns:
            X_expanded.drop(columns=['1'], inplace=True)
        
        # Feature Scaling (Standardization)
        self.scaler = StandardScaler()
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X_expanded), columns=X_expanded.columns)
        X_scaled.insert(0, "Weekday/Weekend", train['Weekday/Weekend'])

        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function is used in testing/inference to transform test data using the fitted polynomial features and scaler from training.
        Args:
            df: DataFrame to transform.
        Returns:
            Transformed DataFrame.
        """
        if self.poly is None or self.scaler is None:
            # If trained polynomial feature transformer or scaler not present, raise error
            raise RuntimeError("DataPreprocessor must be fitted before calling transform.")
        
        df = df.copy()
        # Add date features
        df = self._add_date_features(df)

        # Split features from labels
        X = df[self.feature_columns]

        # Polynomial Feature Expansion
        X_expanded = self.poly.transform(X)
        X_expanded = pd.DataFrame(X_expanded, columns=self.poly.get_feature_names_out())
        if '1' in X_expanded.columns:
            X_expanded.drop(columns=['1'], inplace=True)
        
        # Feature Scaling (Standardization)
        X_scaled = pd.DataFrame(self.scaler.transform(X_expanded), columns=X_expanded.columns)
        X_scaled.insert(0, "Weekday/Weekend", df['Weekday/Weekend'])

        return X_scaled

    def _add_date_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 'Weekday/Weekend' feature based on the 'Date' column.
        """
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        df['Weekday/Weekend'] = (df['Date'].dt.day_of_week < 5).astype(int)
        return df
