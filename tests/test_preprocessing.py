import pandas as pd
import numpy as np
from forest_fire_pred.preprocessing import DataPreprocessor

def test_preprocess_single_input_shape():
    # Create dummy train and test data
    train = pd.DataFrame({
        'Date': ['01-06-2012']*10,
        'Temperature': np.random.rand(10)*30,
        'RH': np.random.rand(10)*100,
        'Ws': np.random.rand(10)*20,
        'Rain': np.random.rand(10),
        'FFMC': np.random.rand(10)*100,
        'DMC': np.random.rand(10)*50,
        'DC': np.random.rand(10)*300,
        'ISI': np.random.rand(10)*10,
        'BUI': np.random.rand(10)*40,
        'Classes': np.random.randint(0, 2, 10)
    })
    test = train.copy()
    preprocessor = DataPreprocessor(train, test)
    X_train, X_test, y_train, y_test = preprocessor.preprocess_data()
    # Prepare a single input row
    single_row = pd.DataFrame({
        'Temperature': [25],
        'RH': [45],
        'Ws': [10],
        'Rain': [0],
        'FFMC': [85],
        'DMC': [20],
        'DC': [200],
        'ISI': [5],
        'BUI': [30],
        'Weekday/Weekend': [1]
    })
    X_single = preprocessor.preprocess_single_input(single_row)
    # Should have same number of columns as X_train
    assert X_single.shape[1] == X_train.shape[1]
    assert X_single.shape[0] == 1

def test_preprocess_single_input_error():
    # Should raise if preprocessor is not fitted
    train = pd.DataFrame({
        'Date': ['01-06-2012']*10,
        'Temperature': np.random.rand(10)*30,
        'RH': np.random.rand(10)*100,
        'Ws': np.random.rand(10)*20,
        'Rain': np.random.rand(10),
        'FFMC': np.random.rand(10)*100,
        'DMC': np.random.rand(10)*50,
        'DC': np.random.rand(10)*300,
        'ISI': np.random.rand(10)*10,
        'BUI': np.random.rand(10)*40,
        'Classes': np.random.randint(0, 2, 10)
    })
    test = train.copy()
    preprocessor = DataPreprocessor(train, test)
    single_row = pd.DataFrame({
        'Temperature': [25],
        'RH': [45],
        'Ws': [10],
        'Rain': [0],
        'FFMC': [85],
        'DMC': [20],
        'DC': [200],
        'ISI': [5],
        'BUI': [30],
        'Weekday/Weekend': [1]
    })
    import pytest
    with pytest.raises(RuntimeError):
        preprocessor.preprocess_single_input(single_row) 