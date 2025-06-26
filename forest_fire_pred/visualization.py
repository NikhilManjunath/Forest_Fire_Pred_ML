"""
Plotting Functions
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from forest_fire_pred.metrics import build_confusion_matrix
from typing import Any

def plot_confusion_matrix(y_true: Any, y_pred: Any) -> None:
    """
    Plot a confusion matrix using seaborn heatmap.
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be of the same length.")
    
    cm = build_confusion_matrix(np.array(y_true), np.array(y_pred))
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show() 