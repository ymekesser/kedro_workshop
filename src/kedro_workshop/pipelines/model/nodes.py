"""
Model pipeline for training and evaluating linear regression on HDB resale prices.
"""

import logging
from typing import Dict, Tuple

import pandas as pd
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    root_mean_squared_error,
)

from .data_utils import create_train_test_split, prepare_features_and_target
from .plot_utils import create_model_analysis_figure

logger = logging.getLogger(__name__)

# Constants (TODO: move to parameters)
TEST_SIZE = 0.2
RANDOM_STATE = 42
FIGURE_SIZE = (15, 12)


def train_linear_regression(feature_set: pd.DataFrame) -> Tuple[LinearRegression, Dict]:
    """Train a linear regression model on the feature set.

    This function:
    1. Separates features from target variable
    2. Splits data into training and test sets
    3. Trains a linear regression model
    4. Evaluates performance on both sets
    5. Returns model and comprehensive evaluation metrics

    Args:
        feature_set: ML-ready dataset with features and target

    Returns:
        Tuple of (trained_model, model_info_dict)
    """
    logger.info("Training linear regression model")

    # Prepare data using utilities
    X, y = prepare_features_and_target(feature_set)
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logger.info(f"Training set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    logger.info(f"Features: {list(X.columns)}")

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate metrics
    train_metrics = {
        "r2": r2_score(y_train, y_train_pred),
        "mse": mean_squared_error(y_train, y_train_pred),
        "mae": mean_absolute_error(y_train, y_train_pred),
        "rmse": root_mean_squared_error(y_train, y_train_pred),
    }

    test_metrics = {
        "r2": r2_score(y_test, y_test_pred),
        "mse": mean_squared_error(y_test, y_test_pred),
        "mae": mean_absolute_error(y_test, y_test_pred),
        "rmse": root_mean_squared_error(y_test, y_test_pred),
    }

    # Feature importance (coefficients)
    # Note: Features are currently not normalized, so the importance may be misleading.
    feature_importance = {
        feature: coef for feature, coef in zip(X.columns, model.coef_)
    }

    model_info = {
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "feature_importance": feature_importance,
        "intercept": model.intercept_,
        "n_features": len(X.columns),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
    }

    logger.info(f"Model trained successfully!")
    logger.info(f"Train R²: {train_metrics['r2']:.4f}")
    logger.info(f"Test R²: {test_metrics['r2']:.4f}")
    logger.info(f"Test RMSE: ${test_metrics['rmse']:,.0f}")

    return model, model_info


def evaluate_model_performance(model_info: Dict) -> pd.DataFrame:
    """Create a summary report of model performance."""
    logger.info("Creating model performance report")

    # Extract metrics
    train_metrics = model_info["train_metrics"]
    test_metrics = model_info["test_metrics"]

    # Create performance summary
    performance_data = [
        {
            "Metric": "R² Score",
            "Train": train_metrics["r2"],
            "Test": test_metrics["r2"],
        },
        {
            "Metric": "RMSE ($)",
            "Train": train_metrics["rmse"],
            "Test": test_metrics["rmse"],
        },
        {
            "Metric": "MAE ($)",
            "Train": train_metrics["mae"],
            "Test": test_metrics["mae"],
        },
        {"Metric": "MSE", "Train": train_metrics["mse"], "Test": test_metrics["mse"]},
    ]

    performance_df = pd.DataFrame(performance_data)

    logger.info("Model Performance Summary:")
    logger.info(f"Test R² Score: {test_metrics['r2']:.4f}")
    logger.info(f"Test RMSE: ${test_metrics['rmse']:,.0f}")
    logger.info(f"Test MAE: ${test_metrics['mae']:,.0f}")

    return performance_df


def create_model_plots(model: LinearRegression, feature_set: pd.DataFrame) -> Figure:
    """Create comprehensive visualization plots for the linear regression model.

    Generates a 2x2 grid of plots to analyze model performance:
    1. Actual vs Predicted scatter plot
    2. Residuals plot to check for patterns
    3. Feature importance from coefficients
    4. Distribution comparison of actual vs predicted

    Args:
        model: Trained linear regression model
        feature_set: Same dataset used for training

    Returns:
        Matplotlib figure with comprehensive model analysis
    """
    logger.info("Creating model visualization plots")

    # Prepare data using same utilities as training
    X, y = prepare_features_and_target(feature_set)
    X_train, X_test, y_train, y_test = create_train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Make predictions on test set
    y_test_pred = model.predict(X_test)

    # Create comprehensive figure using plot utilities
    fig = create_model_analysis_figure(
        y_actual=y_test,
        y_predicted=y_test_pred,
        feature_names=X.columns,
        coefficients=model.coef_,
        figure_size=FIGURE_SIZE,
    )

    logger.info("Model visualization plots created successfully")
    return fig
