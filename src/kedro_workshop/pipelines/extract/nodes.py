import logging
from datetime import datetime
from typing import Any, Dict

import pandas as pd
import requests

logger = logging.getLogger(__name__)


def extract_hdb_resale_prices(hdb_resale_prices: pd.DataFrame) -> pd.DataFrame:
    """Extract HDB resale prices with educational logging and validation."""
    logger.info(f"Extracted {len(hdb_resale_prices)} HDB resale records")

    # Basic validation
    if hdb_resale_prices.empty:
        raise ValueError("HDB resale data is empty")

    expected_columns = ["town", "resale_price", "flat_type", "floor_area_sqm"]
    _validate_expected_columns(hdb_resale_prices, expected_columns, "HDB resale prices")

    return hdb_resale_prices


def extract_mrt_stations(mrt_stations: pd.DataFrame) -> pd.DataFrame:
    """Extract MRT station data with educational logging and validation."""
    logger.info(f"Extracted {len(mrt_stations)} MRT station records")

    # Basic validation
    if mrt_stations.empty:
        raise ValueError("MRT stations data is empty")

    expected_columns = ["Name", "Line", "Code"]
    _validate_expected_columns(mrt_stations, expected_columns, "MRT stations")

    return mrt_stations


def extract_mrt_geodata(mrt_geodata: requests.Response) -> Dict[str, Any]:
    """Extract MRT station geodata from Overpass API response with validation."""
    try:
        # Check if request was successful
        if mrt_geodata.status_code != 200:
            raise ValueError(
                f"API request failed with status {mrt_geodata.status_code}: "
                f"{mrt_geodata.text}"
            )

        data = mrt_geodata.json()

        # Structural Validation
        if not isinstance(data, dict):
            raise ValueError("API response is not a valid JSON object")

        if "elements" not in data:
            raise ValueError("API response missing 'elements' field")

        elements = data.get("elements", [])
        logger.info(f"Extracted {len(elements)} MRT station records")

        # Basic validation - check if we have reasonable data
        if len(elements) == 0:
            logger.warning("No MRT station data found in API response")
        elif len(elements) < 50:  # Singapore has ~200+ MRT stations
            logger.warning(f"Only {len(elements)} MRT stations found - this seems low")

        # Add extraction metadata
        data["extraction_metadata"] = {
            "extracted_at": datetime.now().isoformat(),
            "source": "Overpass API",
            "record_count": len(elements),
            "status_code": mrt_geodata.status_code,
        }

        return data

    except requests.exceptions.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}")
    except Exception as e:
        logger.error(f"Error extracting MRT geodata: {e}")
        raise


def extract_mall_geodata(mall_geodata: requests.Response) -> Dict[str, Any]:
    """Extract shopping mall geodata from Overpass API response with validation."""
    try:
        # Check if request was successful
        if mall_geodata.status_code != 200:
            raise ValueError(
                f"API request failed with status {mall_geodata.status_code}: "
                f"{mall_geodata.text}"
            )

        data = mall_geodata.json()

        # Structural Validation
        if not isinstance(data, dict):
            raise ValueError("API response is not a valid JSON object")

        if "elements" not in data:
            raise ValueError("API response missing 'elements' field")

        elements = data.get("elements", [])
        logger.info(f"Extracted {len(elements)} shopping mall records")

        # Basic validation - check if we have reasonable data
        if len(elements) == 0:
            logger.warning("No shopping mall data found in API response")
        elif len(elements) < 10:  # Singapore should have many malls
            logger.warning(
                f"Only {len(elements)} shopping malls found - this seems low"
            )

        # Log breakdown of mall types if available
        mall_types = {}
        for element in elements:
            element_type = element.get("type", "unknown")
            mall_types[element_type] = mall_types.get(element_type, 0) + 1

        if mall_types:
            logger.info(f"Mall data breakdown: {mall_types}")

        # Add extraction metadata
        data["extraction_metadata"] = {
            "extracted_at": datetime.now().isoformat(),
            "source": "Overpass API",
            "record_count": len(elements),
            "status_code": mall_geodata.status_code,
            "element_types": mall_types,
        }

        return data

    except requests.exceptions.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON response: {e}")
    except Exception as e:
        logger.error(f"Error extracting mall geodata: {e}")
        raise


def extract_hdb_address_geodata(hdb_address_geodata: pd.DataFrame) -> pd.DataFrame:
    """Extract HDB address geodata with educational logging and validation."""
    logger.info(f"Extracted {len(hdb_address_geodata)} HDB address records")
    logger.info(f"Data shape: {hdb_address_geodata.shape}")

    # Basic validation
    if hdb_address_geodata.empty:
        raise ValueError("HDB address geodata is empty")

    expected_columns = ["latitude", "longitude"]
    _validate_expected_columns(
        hdb_address_geodata, expected_columns, "HDB address geodata"
    )

    return hdb_address_geodata


def _validate_expected_columns(
    data: pd.DataFrame, expected_columns: list[str], dataset_name: str
) -> None:
    """Helper function to validate expected columns are present in the dataset."""
    missing_columns = [col for col in expected_columns if col not in data.columns]
    if missing_columns:
        logger.warning(f"Missing expected columns in {dataset_name}: {missing_columns}")
