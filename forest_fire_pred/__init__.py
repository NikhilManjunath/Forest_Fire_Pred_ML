"""
Forest Fire Prediction ML package

This package provides tools for loading, preprocessing, modeling, evaluating, and visualizing forest fire prediction data and results.
"""

from .data_loader import Dataloader
from .preprocessing import DataPreprocessor
from .models import ModelTrainer, Experiments
from .metrics import compute_metrics, build_confusion_matrix
from .visualization import plot_confusion_matrix
from . import utils
