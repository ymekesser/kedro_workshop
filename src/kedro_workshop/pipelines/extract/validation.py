"""Validation utilities for data extraction pipeline.

This module contains reusable validation functions that help ensure
data quality across different extraction nodes. These utilities
demonstrate common data validation patterns in data engineering.
"""

import logging
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def validate_dataframe_not_empty(df: pd.DataFrame, dataset_name: str) -> None:
    """Check if dataframe contains data.

    Args:
        df: DataFrame to validate
        dataset_name: Human-readable name for logging

    Raises:
        ValueError: If dataframe is empty
    """
    if df.empty:
        raise ValueError(f"{dataset_name} data is empty")


def validate_required_columns(
    df: pd.DataFrame, required_columns: list[str], dataset_name: str
) -> None:
    """Check if all required columns are present in the dataframe.

    Args:
        df: DataFrame to validate
        required_columns: List of column names that must be present
        dataset_name: Human-readable name for logging

    Note:
        Logs warnings for missing columns but doesn't raise exceptions,
        allowing pipeline to continue with available data.
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.warning(f"Missing expected columns in {dataset_name}: {missing_columns}")


def validate_api_response_success(response: requests.Response) -> None:
    """Check if API request was successful (Status 200).

    Args:
        response: HTTP response object

    Raises:
        ValueError: If response status indicates failure
    """
    if response.status_code != 200:
        raise ValueError(
            f"API request failed with status {response.status_code}: "
            f"{response.text}"
        )


def validate_overpass_data_structure(data: Any) -> None:
    """Check if Overpass API response has expected structure.

    Args:
        data: Parsed JSON response from Overpass API

    Raises:
        ValueError: If data structure is invalid

    Note:
        Overpass API returns data in a specific format with an 'elements' array.
        This validates the basic structure before processing.
    """
    if not isinstance(data, dict):
        raise ValueError("API response is not a valid JSON object")

    if "elements" not in data:
        raise ValueError("API response missing 'elements' field")


def validate_data_quantity(
    element_count: int, expected_minimum: int, data_type: str
) -> None:
    """Log warnings about potentially low data counts.

    Args:
        element_count: Number of elements found
        expected_minimum: Minimum expected count
        data_type: Type of data for logging context

    Note:
        Real-world pipelines should have monitoring based on historical patterns.
    """
    if element_count == 0:
        logger.warning(f"No {data_type} data found in API response")
    elif element_count < expected_minimum:
        logger.warning(
            f"Only {element_count} {data_type} found - this seems low for Singapore"
        )
