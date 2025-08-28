"""
Model pipeline for training and evaluating linear regression on HDB resale prices.
"""

from kedro.pipeline import Node, Pipeline, node

from .nodes import train_linear_regression, evaluate_model_performance, create_model_plots


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        node(
            func=train_linear_regression,
            inputs="transformed_feature_set",
            outputs=["linear_regression_model", "model_info"],
            name="train_linear_regression_node",
        ),
        node(
            func=evaluate_model_performance,
            inputs="model_info",
            outputs="model_performance_report",
            name="evaluate_model_performance_node",
        ),
        node(
            func=create_model_plots,
            inputs=["linear_regression_model", "transformed_feature_set"],
            outputs="model_plots",
            name="create_model_plots_node",
        ),
    ])
