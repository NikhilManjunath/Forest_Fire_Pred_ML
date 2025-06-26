import pytest
import pandas as pd
import numpy as np
from forest_fire_pred.preprocessing import DataPreprocessor

@pytest.fixture
def sample_df():
    data = {
        'Date': ['27/06/2025', '28/06/2025', '29/06/2025'],  # Fri, Sat, Sun
        'Temperature': [25, 30, 22],
        'RH': [23, 50, 34],
        'Ws': [3, 1, 5],
        'Rain': [0.0, 0.4, 0.0],
        'FFMC': [85, 90, 78],
        'DMC': [20, 25, 30],
        'DC': [100, 43.4, 77.8],
        'ISI': [5.0, 6.0, 7.0],
        'BUI': [30.0, 35.0, 32.0],
        'Classes': [0, 1, 0]
    }
    return pd.DataFrame(data)

def test_add_date_feature(sample_df):
    preprocessor = DataPreprocessor()
    result = preprocessor._add_date_features(sample_df.copy())
    assert 'Weekday/Weekend' in result.columns
    assert result['Weekday/Weekend'].tolist() == [1, 0, 0]  # Fri = 1, Sat/Sun = 0

def test_preprocessing_fit(sample_df):
    preprocessor = DataPreprocessor()
    X, y = preprocessor.fit(sample_df)
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert 'Weekday/Weekend' in X.columns
    assert X.shape[0] == sample_df.shape[0]
    assert y.equals(sample_df['Classes'])
    assert '1' not in X.columns  # Bias term should be removed after poly
    

def test_transform(sample_df):
    preprocessor = DataPreprocessor()
    X_train, _ = preprocessor.fit(sample_df)
    X_test = preprocessor.transform(sample_df)
    assert list(X_train.columns) == list(X_test.columns)

def test_transform_fail(sample_df):
    preprocessor = DataPreprocessor()   # Poly and Scaler not initialized
    with pytest.raises(RuntimeError):
        preprocessor.transform(sample_df)
