"""
Helper functions for feature selection and cross-validation.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from typing import Tuple, List, Dict, Any

def PCA_func(train: pd.DataFrame, n_components: int) -> Tuple[PCA, pd.DataFrame, pd.DataFrame]:
    """
    Fit  PCA to data and return transformed DataFrame along with the PCA model.
    Args:
        train: Training DataFrame.
        n_components: Number of components to keep.
    Returns:
        Tuple of (PCA object, transformed train DataFrame).
    """

    if n_components > train.shape[1]:
        raise ValueError(f"n_components ({n_components}) cannot be greater than number of features ({train.shape[1]})")
    
    pca = PCA(n_components=n_components)
    train_PCA = pca.fit_transform(train)
    train_PCA = pd.DataFrame(train_PCA)

    return pca, train_PCA

def seq_feat_selector(train: pd.DataFrame, labels: pd.Series, n_max: int) -> List[str]:
    """
    Sequential feature selection using multiple models and cross-validation.
    Args:
        train: Training DataFrame.
        labels: Training labels.
        n_max: Maximum number of features to select.
    Returns:
        List of selected feature column names.
    """

    if n_max > train.shape[1]:
        raise ValueError(f"n_max ({n_max}) cannot be greater than number of features ({train.shape[1]})")
    
    # Use cross-validation to evaluate
    folds, classes = k_fold(train, labels)
    train_features_folds, train_classes_folds, _, _ = cross_val(folds, classes)

    # Models to experiment with
    knn = KNeighborsClassifier()
    sv_c = SVC(kernel='linear')
    ridge = RidgeClassifier()
    logistic = LogisticRegression(penalty='l2')

    feat_model = []
    for model in [knn, sv_c, ridge, logistic]:
        feat_per_fold = []
        for i in range(len(train_classes_folds)):
            sfs = SequentialFeatureSelector(model, n_features_to_select=18)
            sfs.fit(train_features_folds[i], train_classes_folds[i])
            feat_per_fold.append(sfs.get_support(indices=True))
        feat_model.append(feat_per_fold)
    unique, counts = np.unique(feat_model, return_counts=True)
    res = unique[np.argsort(counts)]
    cols = res[-n_max:]

    return train.columns[cols].tolist()

def k_fold(features: pd.DataFrame, labels: pd.Series) -> Tuple[Dict[int, pd.DataFrame], Dict[int, pd.Series]]:
    """
    Split features and labels into k=6 folds for cross-validation.
    Args:
        features: Feature DataFrame.
        labels: Label Series.
    Returns:
        Tuple of (folds dictionary, class_folds dictionary).
    """

    folds = {}
    class_folds = {}
    for i in range(0, 6):
        folds[i] = features[30 * i:30 * (i + 1)]
        class_folds[i] = labels[30 * i:30 * (i + 1)]
    return folds, class_folds

def cross_val(folds: Dict[int, pd.DataFrame], classes: Dict[int, pd.Series]) -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    """
    Generate train/validation splits for each fold.
    Args:
        folds: Dictionary of feature folds.
        classes: Dictionary of class folds.
    Returns:
        Tuple of (train_features_folds, train_classes_folds, val_features_folds, val_classes_folds).
    """

    train_features_folds = []
    val_features_folds = []
    train_classes_folds = []
    val_classes_folds = []

    keys = set(folds.keys())
    for i in range(len(folds)):
        val = folds[i].copy(deep=True)
        val_classes = classes[i].copy(deep=True)
        val.drop(val.tail(6).index, inplace=True)
        val_classes.drop(val_classes.tail(6).index, inplace=True)
        train = []
        train_class = []
        if i != 0:
            excludes = set([i, i - 1])
            val_prev = folds[i - 1].copy(deep=True)
            val_prev.drop(val_prev.tail(6).index, inplace=True)
            val_prev_class = classes[i - 1].copy(deep=True)
            val_prev_class.drop(val_prev_class.tail(6).index, inplace=True)
            train.append(val_prev)
            train_class.append(val_prev_class)
        else:
            excludes = set([i])
        for key in keys.difference(excludes):
            train.append(folds[key])
            train_class.append(classes[key])
        train = np.vstack(train)
        train_class = np.hstack(train_class)
        train_features_folds.append(train)
        train_classes_folds.append(train_class)
        val_features_folds.append(val)
        val_classes_folds.append(val_classes)
    return train_features_folds, train_classes_folds, val_features_folds, val_classes_folds 