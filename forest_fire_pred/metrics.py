"""
Evaluation Metrics Helper Functions
"""
import numpy as np
from typing import Any, Dict
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
    """
    Generate classification metrics - accuracy, precision, recall, F1 score, and False Negatives.
    Args:
        preds: Predicted labels as a numpy array.
        labels: True labels as a numpy array.
    Returns:
        Dictionary with metrics: false_negatives, accuracy, precision, recall, f1_score.
    """
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        'false_negatives': fn,
        'accuracy': round(accuracy_score(labels, preds), 2),
        'precision': round(precision_score(labels, preds, zero_division=0), 2),
        'recall': round(recall_score(labels, preds, zero_division=0), 2),
        'f1_score': round(f1_score(labels, preds, zero_division=0), 2)
    }
    return metrics

def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Build a confusion matrix from true and predicted labels.
    Args:
        y_true: True labels as a numpy array.
        y_pred: Predicted labels as a numpy array.
    Returns:
        Confusion matrix as a numpy array.
    """
    return confusion_matrix(y_true, y_pred)