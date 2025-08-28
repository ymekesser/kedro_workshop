"""Data preparation utilities for machine learning pipeline.

Contains reusable functions for preparing data for training and evaluation,
ensuring consistency across different nodes in the pipeline.
"""

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_features_and_target(feature_set: pd.DataFrame, target_column: str = 'resale_price') -> Tuple[pd.DataFrame, pd.Series]:
    """Separate features and target variable from feature set.
    
    Args:
        feature_set: Complete dataset with features and target
        target_column: Name of target variable column
        
    Returns:
        Tuple of (features_df, target_series)
    """
    X = feature_set.drop(target_column, axis=1)
    y = feature_set[target_column]
    return X, y


def create_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, 
                           random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create consistent train/test split for model training and evaluation.
    
    Uses the same random state across all functions to ensure consistency
    between training and plotting/evaluation.
    
    Args:
        X: Feature dataframe
        y: Target series
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducible splits
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)