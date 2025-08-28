"""Shared utilities for data cleaning operations.

Contains only the reusable functions that are used by multiple
cleaning nodes.
"""

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def convert_columns_to_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Convert specified columns to numeric format, handling errors gracefully.

    Args:
        df: DataFrame to process
        columns: List of column names to convert to numeric

    Returns:
        DataFrame with specified columns converted to numeric
    """
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
    return df_copy


def convert_overpass_json_to_dataframe(data: Any) -> pd.DataFrame:
    """Convert Overpass API JSON response to pandas DataFrame.

    Args:
        data: JSON response from Overpass API

    Returns:
        DataFrame with flattened structure from JSON elements
    """
    data_dict = pd.json_normalize(data, record_path=["elements"])
    return pd.DataFrame.from_dict(data_dict, orient="columns")  # type: ignore


def process_overpass_geodata_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Process Overpass API geodata to standardize coordinate columns.

    Overpass API returns coordinates in different formats:
    - Direct lat/lon fields for points
    - center.lat/center.lon for complex shapes

    This function coalesces them into standard latitude/longitude columns.

    Args:
        df: DataFrame from Overpass API JSON conversion

    Returns:
        DataFrame with standardized latitude/longitude columns
    """
    # Coalesce lat/center.lat and lon/center.lon
    center_lat = df.get("center.lat", pd.Series(index=df.index, dtype="float64"))
    center_lon = df.get("center.lon", pd.Series(index=df.index, dtype="float64"))

    df["latitude"] = df["lat"].fillna(center_lat)
    df["longitude"] = df["lon"].fillna(center_lon)

    return df


def handle_geodata_duplicates(df: pd.DataFrame, location_type: str) -> pd.DataFrame:
    """Handle duplicate location names by averaging their coordinates.

    Assumes that locations with the same name are clustered together
    geographically and takes the mean of their coordinates.

    Args:
        df: DataFrame with columns ['name', 'latitude', 'longitude']
        location_type: Type of location for logging (e.g., "MRT", "mall")

    Returns:
        DataFrame with duplicates merged by taking mean coordinates
    """
    if len(df) == 0:
        return df

    cleaned_df = (
        df.groupby("name").agg({"latitude": "mean", "longitude": "mean"}).reset_index()
    )

    duplicates_removed = len(df) - len(cleaned_df)
    if duplicates_removed > 0:
        logger.info(f"Merged {duplicates_removed} duplicate {location_type} locations")

    logger.info(f"Final {location_type} geodata: {len(cleaned_df)} unique locations")
    return cleaned_df
