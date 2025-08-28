"""Plotting utilities for model visualization and analysis.

Contains individual plotting functions that can be composed together
to create comprehensive model analysis visualizations.

For workshop participants: Each function creates one specific plot,
making it easier to understand what each visualization shows.
"""

import logging
from typing import Tuple

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

logger = logging.getLogger(__name__)


def create_actual_vs_predicted_plot(
    ax: Axes, y_actual: pd.Series, y_predicted: np.ndarray
) -> None:
    """Create scatter plot comparing actual vs predicted values.

    This plot helps visualize how well the model predictions match reality.
    Points close to the diagonal line indicate good predictions.

    Args:
        ax: Matplotlib axes object to plot on
        y_actual: Series of actual target values
        y_predicted: Array of predicted values
    """
    ax.scatter(y_actual, y_predicted, alpha=0.5, s=1)

    # Add perfect prediction line (diagonal)
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "r--", lw=2, label="Perfect Prediction"
    )

    ax.set_xlabel("Actual Price ($)")
    ax.set_ylabel("Predicted Price ($)")
    ax.set_title("Actual vs Predicted Prices")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style="plain", axis="both")


def create_residuals_plot(
    ax: Axes, y_actual: pd.Series, y_predicted: np.ndarray
) -> None:
    """Create residuals plot to check for patterns in prediction errors.

    Residuals should be randomly scattered around zero. Patterns in residuals
    indicate the model might be missing important relationships.

    Args:
        ax: Matplotlib axes object to plot on
        y_actual: Series of actual target values
        y_predicted: Array of predicted values
    """
    residuals = y_actual - y_predicted
    ax.scatter(y_predicted, residuals, alpha=0.5, s=1)
    ax.axhline(y=0, color="r", linestyle="--", lw=2)

    ax.set_xlabel("Predicted Price ($)")
    ax.set_ylabel("Residuals ($)")
    ax.set_title("Residuals vs Predicted")
    ax.grid(True, alpha=0.3)


def create_feature_importance_plot(
    ax: Axes, feature_names: pd.Index, coefficients: np.ndarray
) -> None:
    """Create horizontal bar chart showing feature importance from model coefficients.

    Longer bars indicate features that have more impact on predictions.
    Red bars show negative impact, blue bars show positive impact.

    Args:
        ax: Matplotlib axes object to plot on
        feature_names: Names of features
        coefficients: Linear regression coefficients for each feature
    """
    # Create DataFrame for easy sorting
    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "abs_coefficient": abs(coefficients),
        }
    ).sort_values("abs_coefficient", ascending=True)

    # Color bars based on positive/negative coefficients
    colors = [
        "red" if coef < 0 else "blue" for coef in feature_importance["coefficient"]
    ]
    bars = ax.barh(
        feature_importance["feature"],
        feature_importance["coefficient"],
        color=colors,
        alpha=0.7,
    )

    ax.set_xlabel("Coefficient Value ($)")
    ax.set_title("Feature Importance (Coefficients)")
    ax.grid(True, alpha=0.3, axis="x")

    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        # Position text slightly offset from bar end
        offset = (max(coefficients) - min(coefficients)) * 0.01
        ax.text(
            width + offset,
            bar.get_y() + bar.get_height() / 2,
            f"${width:,.0f}",
            ha="left" if width >= 0 else "right",
            va="center",
            fontsize=9,
        )


def create_distribution_comparison_plot(
    ax: Axes, y_actual: pd.Series, y_predicted: np.ndarray
) -> None:
    """Create histogram comparing actual vs predicted value distributions.

    The distributions should be similar if the model is capturing
    the overall pattern in the data well.

    Args:
        ax: Matplotlib axes object to plot on
        y_actual: Series of actual target values
        y_predicted: Array of predicted values
    """
    ax.hist(y_actual, bins=50, alpha=0.7, label="Actual", density=True, color="blue")
    ax.hist(
        y_predicted, bins=50, alpha=0.7, label="Predicted", density=True, color="red"
    )

    ax.set_xlabel("Price ($)")
    ax.set_ylabel("Density")
    ax.set_title("Price Distribution: Actual vs Predicted")
    ax.legend()
    ax.grid(True, alpha=0.3)


def add_metrics_text_box(
    fig: Figure, y_actual: pd.Series, y_predicted: np.ndarray
) -> None:
    """Add model performance metrics as a text box on the figure.

    Args:
        fig: Matplotlib figure object
        y_actual: Series of actual target values
        y_predicted: Array of predicted values
    """
    # Calculate metrics
    r2 = r2_score(y_actual, y_predicted)
    rmse = root_mean_squared_error(y_actual, y_predicted)
    mae = mean_absolute_error(y_actual, y_predicted)

    # Create formatted text
    metrics_text = f"R² = {r2:.3f}\nRMSE = ${rmse:,.0f}\nMAE = ${mae:,.0f}"

    # Add text box in bottom-left corner
    fig.text(
        0.02,
        0.02,
        metrics_text,
        fontsize=12,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
    )


def create_model_analysis_figure(
    y_actual: pd.Series,
    y_predicted: np.ndarray,
    feature_names: pd.Index,
    coefficients: np.ndarray,
    figure_size: Tuple[int, int] = (15, 12),
) -> Figure:
    """Create comprehensive model analysis figure with multiple plots.

    This is the main function that combines all individual plots into
    a single comprehensive visualization.

    Args:
        y_actual: Series of actual target values
        y_predicted: Array of predicted values
        feature_names: Names of model features
        coefficients: Linear regression coefficients
        figure_size: Tuple of (width, height) for figure size

    Returns:
        Complete matplotlib figure with all analysis plots
    """
    # Create figure with 2x2 subplot grid
    fig, axes = plt.subplots(2, 2, figsize=figure_size)
    fig.suptitle("Linear Regression Model Analysis", fontsize=16, fontweight="bold")

    # Create each individual plot
    create_actual_vs_predicted_plot(axes[0, 0], y_actual, y_predicted)
    create_residuals_plot(axes[0, 1], y_actual, y_predicted)
    create_feature_importance_plot(axes[1, 0], feature_names, coefficients)
    create_distribution_comparison_plot(axes[1, 1], y_actual, y_predicted)

    # Add metrics text box
    add_metrics_text_box(fig, y_actual, y_predicted)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.12)

    return fig
