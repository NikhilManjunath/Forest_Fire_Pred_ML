import numpy as np
import pandas as pd
from forest_fire_pred import PCA_func, seq_feat_selector
import pytest

@pytest.mark.parametrize("n_components", [1, 2, 5])
def test_pca_func(n_components):
    df = pd.DataFrame(np.random.rand(10, 5))
    pca, transformed = PCA_func(df, n_components)
    assert transformed.shape == (10, n_components)
    assert pca.n_components == n_components

@pytest.mark.parametrize("n_components", [6, 10])
def test_pca_func_invalid(n_components):
    df = pd.DataFrame(np.random.rand(10, 5))
    with pytest.raises(ValueError):
        PCA_func(df, n_components)

@pytest.mark.parametrize("n_max", [2, 4, 6])
def test_seq_feat_selector(n_max):
    X = pd.DataFrame(np.random.rand(180, 20))
    y = pd.Series(np.random.randint(0, 2, 180))
    selected = seq_feat_selector(X, y, n_max=n_max)
    assert isinstance(selected, list)
    assert len(selected) == n_max

@pytest.mark.parametrize("n_max", [21, 30])
def test_seq_feat_selector_parametrize_invalid(n_max):
    X = pd.DataFrame(np.random.rand(180, 20))
    y = pd.Series(np.random.randint(0, 2, size=180))
    with pytest.raises(ValueError):
        seq_feat_selector(X, y, n_max=n_max)