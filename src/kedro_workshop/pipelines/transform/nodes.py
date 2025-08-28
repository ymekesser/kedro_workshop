"""
Transform pipeline for creating ML feature set with HDB resale prices and distance features.
"""

import logging

import pandas as pd

from .geo_utils import find_nearest_locations

logger = logging.getLogger(__name__)


def create_feature_set(
    hdb_resale_prices: pd.DataFrame,
    hdb_addresses: pd.DataFrame,
    mrt_stations: pd.DataFrame,
    mrt_geodata: pd.DataFrame,
    mall_geodata: pd.DataFrame,
) -> pd.DataFrame:
    """Create machine learning ready feature set with distance-based features.

    This function combines HDB resale transaction data with geographic proximity
    features by calculating distances to the nearest MRT station and shopping mall
    for each property address.

    The process:
    1. Filters MRT geodata to include only operational stations
    2. Calculates nearest MRT and mall distances for each HDB address
    3. Joins all data together on address (block + street_name)
    4. Selects ML-ready numeric features and removes incomplete records

    Args:
        hdb_resale_prices: Clean HDB transaction data with derived features
        hdb_addresses: Clean HDB address coordinates
        mrt_stations: Clean operational MRT station reference data
        mrt_geodata: Clean MRT station geographic coordinates
        mall_geodata: Clean shopping mall geographic coordinates

    Note:
        Only returns complete records (no missing values) ready for ML modeling.
    """
    logger.info("Creating ML feature set with distance features")

    # Filter MRT geodata to only include operational stations
    operational_mrt = mrt_geodata[mrt_geodata["name"].isin(mrt_stations["Name"])]
    logger.info(
        f"Filtered to {len(operational_mrt)} operational MRT stations (from {len(mrt_geodata)} total)"
    )

    # Find nearest MRT and Mall for each address
    mrt_distances = find_nearest_locations(hdb_addresses, operational_mrt, "mrt")
    mall_distances = find_nearest_locations(hdb_addresses, mall_geodata, "mall")

    # Join distance features
    address_features = mrt_distances.merge(
        mall_distances, on=["block", "street_name"], how="inner"
    )

    # Join with HDB resale prices
    feature_set = hdb_resale_prices.merge(
        address_features, on=["block", "street_name"], how="inner"
    )

    # Select ML-ready features
    ml_features = [
        "resale_price",  # target
        "floor_area_sqm",
        "room_count",
        "remaining_lease_months",
        "storey_median",
        "nearest_mrt_distance_km",
        "nearest_mall_distance_km",
    ]

    # Keep only complete cases
    feature_set = feature_set[ml_features].dropna()

    logger.info(
        f"Final feature set: {len(feature_set)} records with {len(ml_features)} features"
    )
    logger.info(f"Features: {ml_features}")

    return feature_set
