import numpy as np
import pandas as pd
from forest_fire_pred.utils import PCA_func, seq_feat_selector

def test_pca_func():
    train = pd.DataFrame(np.random.rand(10, 5))
    test = pd.DataFrame(np.random.rand(5, 5))
    train_pca, test_pca = PCA_func(train, test, n_components=2)
    assert train_pca.shape[1] == 2
    assert test_pca.shape[1] == 2
    assert train_pca.shape[0] == 10
    assert test_pca.shape[0] == 5

def test_seq_feat_selector():
    # 180 samples, 20 features, binary labels
    X = pd.DataFrame(np.random.rand(180, 20), columns=[f'f{i}' for i in range(20)])
    y = pd.Series(np.random.randint(0, 2, 180))
    selected = seq_feat_selector(X, y, n_max=5)
    assert isinstance(selected, list)
    assert len(selected) == 5
    for col in selected:
        assert col in X.columns 