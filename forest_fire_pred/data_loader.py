"""
Data Loading Utilities
"""
import pandas as pd
from typing import Optional
import logging

class Dataloader:
    """
    Class to load the forest fires data from a CSV file.
    """
    def __init__(self, data_path: str):
        """
        Initialize the Dataloader with the path to the data file.
        """
        self.data_path = data_path

    def load_data(self) -> Optional[pd.DataFrame]:
        """
        Load data from the CSV file.
        """
        return pd.read_csv(self.data_path)