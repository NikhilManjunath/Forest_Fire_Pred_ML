import subprocess
import sys
import os
import tempfile
import pandas as pd
import numpy as np

def make_temp_csv(data: pd.DataFrame) -> str:
    fd, path = tempfile.mkstemp(suffix='.csv')
    os.close(fd)
    data.to_csv(path, index=False)
    return path

def test_main_predict_single():
    # Create valid train and test data with 180 samples
    df = pd.DataFrame({
        'Date': ['01-06-2012']*180,
        'Temperature': np.random.rand(180)*30,
        'RH': np.random.rand(180)*100,
        'Ws': np.random.rand(180)*20,
        'Rain': np.random.rand(180),
        'FFMC': np.random.rand(180)*100,
        'DMC': np.random.rand(180)*50,
        'DC': np.random.rand(180)*300,
        'ISI': np.random.rand(180)*10,
        'BUI': np.random.rand(180)*40,
        'Classes': np.random.randint(0, 2, 180)
    })
    train_path = make_temp_csv(df)
    test_path = make_temp_csv(df)
    # Valid feature vector for prediction
    features = [25, 45, 10, 0, 85, 20, 200, 5, 30, 1]
    feature_str = ','.join(str(x) for x in features)
    result = subprocess.run([
        sys.executable, 'main.py',
        '--train-data', train_path,
        '--test-data', test_path,
        '--predict-single', feature_str
    ], capture_output=True, text=True)
    os.remove(train_path)
    os.remove(test_path)
    assert result.returncode == 0
    assert 'Prediction:' in result.stdout or 'Prediction:' in result.stderr

def test_main_fails_with_missing_column():
    # Missing 'Classes' column
    df = pd.DataFrame({
        'Date': ['01-06-2012']*10,
        'Temperature': np.random.rand(10)*30,
        'RH': np.random.rand(10)*100,
        'Ws': np.random.rand(10)*20,
        'Rain': np.random.rand(10),
        'FFMC': np.random.rand(10)*100,
        'DMC': np.random.rand(10)*50,
        'DC': np.random.rand(10)*300,
        'ISI': np.random.rand(10)*10,
        'BUI': np.random.rand(10)*40
    })
    train_path = make_temp_csv(df)
    test_path = make_temp_csv(df)
    result = subprocess.run([
        sys.executable, 'main.py',
        '--train-data', train_path,
        '--test-data', test_path
    ], capture_output=True, text=True)
    os.remove(train_path)
    os.remove(test_path)
    assert result.returncode != 0 or 'KeyError' in result.stderr or 'KeyError' in result.stdout 