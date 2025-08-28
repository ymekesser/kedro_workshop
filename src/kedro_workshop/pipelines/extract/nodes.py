"""Extract pipeline nodes for processing raw data sources.

This module contains the extraction nodes that:
1. Load data from various sources (CSV files, Excel files, APIs)
2. Perform basic validation and logging
3. Pass clean data to the next pipeline stage
"""

import logging
from typing import Any, Dict

import pandas as pd
import requests

from .validation import (
    validate_data_quantity,
    validate_api_response_success,
    validate_dataframe_not_empty,
    validate_overpass_data_structure,
    validate_required_columns,
)

logger = logging.getLogger(__name__)


def extract_hdb_resale_prices(hdb_resale_prices: pd.DataFrame) -> pd.DataFrame:
    """Extract and validate HDB resale price data.

    This node processes HDB resale transaction data, checking for:
    - Non-empty dataset
    - Required columns for downstream analysis
    """
    logger.info(f"Processing {len(hdb_resale_prices)} HDB resale records")

    # Validate data quality using shared utilities
    validate_dataframe_not_empty(hdb_resale_prices, "HDB resale prices")

    required_columns = ["town", "resale_price", "flat_type", "floor_area_sqm"]
    validate_required_columns(hdb_resale_prices, required_columns, "HDB resale prices")

    return hdb_resale_prices


def extract_mrt_stations(mrt_stations: pd.DataFrame) -> pd.DataFrame:
    """Extract and validate MRT station reference data.

    This node processes MRT station information, checking for:
    - Non-empty dataset
    - Required columns for station identification
    """
    logger.info(f"Processing {len(mrt_stations)} MRT station records")

    # Validate data quality using shared utilities
    validate_dataframe_not_empty(mrt_stations, "MRT stations")

    required_columns = ["Name", "Line", "Code"]
    validate_required_columns(mrt_stations, required_columns, "MRT stations")

    return mrt_stations


def extract_mrt_geodata(mrt_geodata: requests.Response) -> Dict[str, Any]:
    """Extract MRT station geographic data from Overpass API.

    This node processes API response data containing MRT station locations.
    It demonstrates how to handle external API data with proper validation.
    """
    try:
        # Validate API response using shared utilities
        validate_api_response_success(mrt_geodata)

        # Parse and validate JSON structure
        data = mrt_geodata.json()
        validate_overpass_data_structure(data)

        # Extract elements and log results
        elements = data.get("elements", [])
        logger.info(f"Extracted {len(elements)} MRT station geodata records")

        # Quality checks for workshop demonstration
        validate_data_quantity(len(elements), 50, "MRT stations")

        return data

    except requests.exceptions.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}")
    except Exception as e:
        logger.error(f"Error extracting MRT geodata: {e}")
        raise


def extract_mall_geodata(mall_geodata: requests.Response) -> Dict[str, Any]:
    """Extract shopping mall geographic data from Overpass API.

    This node processes API response data containing shopping mall locations.
    Similar pattern to MRT geodata extraction but for retail locations.
    """
    try:
        # Validate API response using shared utilities
        validate_api_response_success(mall_geodata)

        # Parse and validate JSON structure
        data = mall_geodata.json()
        validate_overpass_data_structure(data)

        # Extract elements and log results
        elements = data.get("elements", [])
        logger.info(f"Extracted {len(elements)} shopping mall geodata records")

        # Quality checks for workshop demonstration
        validate_data_quantity(len(elements), 10, "shopping malls")

        return data

    except requests.exceptions.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}")
    except Exception as e:
        logger.error(f"Error extracting mall geodata: {e}")
        raise


def extract_hdb_address_geodata(hdb_address_geodata: pd.DataFrame) -> pd.DataFrame:
    """Extract and validate HDB address geographic data.

    This node processes HDB address coordinates that will be used
    to calculate distances to amenities like MRT stations and malls.
    """
    logger.info(f"Processing {len(hdb_address_geodata)} HDB address records")
    logger.info(f"Data shape: {hdb_address_geodata.shape}")

    # Validate data quality using shared utilities
    validate_dataframe_not_empty(hdb_address_geodata, "HDB address geodata")

    required_columns = ["latitude", "longitude"]
    validate_required_columns(
        hdb_address_geodata, required_columns, "HDB address geodata"
    )

    return hdb_address_geodata
