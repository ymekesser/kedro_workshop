import logging
from typing import Any

import pandas as pd

from .utils import (
    convert_columns_to_numeric,
    convert_overpass_json_to_dataframe,
    handle_geodata_duplicates,
    process_overpass_geodata_coordinates,
)

logger = logging.getLogger(__name__)

# Constants for data cleaning
GEOCODING_CONFIDENCE_THRESHOLD = 0.8


def clean_hdb_resale_prices(hdb_resale_prices: pd.DataFrame) -> pd.DataFrame:
    """Clean and enhance HDB resale price data with derived features.

    This function performs several cleaning and enhancement operations:
    1. Removes unnecessary columns (like 'Unnamed' columns from CSV)
    2. Converts date and numeric fields to proper data types
    3. Parses complex text fields into structured numeric data:
       - Storey ranges ('04 TO 06') -> min, max, median values
       - Remaining lease ('56 years 09 months') -> total months
       - Flat types ('3 ROOM') -> room count

    These derived features will be useful for analysis and modeling.
    """
    logger.info(f"Cleaning {len(hdb_resale_prices)} HDB resale price records")

    df = hdb_resale_prices

    # Remove any unnamed/extra columns (like 'Column1')
    df = df.loc[
        :, ~df.columns.str.contains(r"^Unnamed|Column\d+$", case=False, na=False)
    ]

    # Convert month to datetime
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], format="%m/%d/%Y", errors="coerce")

    # Convert basic numeric columns
    numeric_columns = ["resale_price", "floor_area_sqm", "lease_commence_date"]
    df = convert_columns_to_numeric(df, numeric_columns)

    # Parse storey range to get min, max, and median storey
    logger.info("Parsing storey ranges...")
    df[["storey_min", "storey_max", "storey_median"]] = df["storey_range"].apply(
        lambda x: pd.Series(_parse_storey_range(x))
    )

    # Parse remaining lease to convert to months
    logger.info("Parsing remaining lease...")
    df["remaining_lease_months"] = df.apply(
        lambda row: _parse_remaining_lease(
            row["remaining_lease"], row.get("lease_commence_date"), row.get("month")
        ),
        axis=1,
    )

    # Extract room count from flat_type
    logger.info("Extracting room count...")
    df["room_count"] = df["flat_type"].apply(_extract_room_count)

    logger.info(f"Cleaned data: {len(df)} records")
    return df


def clean_mrt_stations(mrt_stations: pd.DataFrame) -> pd.DataFrame:
    """Clean MRT station data to include only operational stations.

    This function:
    1. Converts opening dates from text to proper datetime format
    2. Filters out planned/future stations (where dates couldn't be parsed)
    3. Counts the number of lines serving each station from station codes
       (e.g. 'NS1 EW24' indicates 2 lines: North-South and East-West)
    """
    logger.info(f"Cleaning {len(mrt_stations)} MRT station records")

    df = mrt_stations

    # Convert Opening column to datetime
    logger.info("Converting opening dates...")
    df["opening_date"] = pd.to_datetime(
        df["Opening"], format="%m/%d/%Y", errors="coerce"
    )

    # Remove stations where opening date couldn't be parsed (planned stations, e.g. 'mid-2028')
    operational_stations = df.dropna(subset=["opening_date"])
    logger.info(f"Filtered out {len(df) - len(operational_stations)} planned stations")
    df = operational_stations

    # Count number of lines per station from station codes
    logger.info("Counting lines per station...")
    df["line_count"] = df["Code"].apply(_count_mrt_station_lines)

    logger.info(f"Cleaned data: {len(df)} operational stations")
    return df


def clean_mrt_geodata(mrt_geodata: Any) -> pd.DataFrame:
    """Clean MRT geographic data from Overpass API response.

    Converts JSON response from OpenStreetMap's Overpass API into a clean
    DataFrame with station names and coordinates. Handles duplicate entries
    by averaging their coordinates.
    """
    logger.info("Converting MRT geodata from JSON...")
    df = convert_overpass_json_to_dataframe(mrt_geodata)
    logger.info(f"Converted {len(df)} raw MRT geodata records")

    # Process coordinates from different Overpass API formats
    df = process_overpass_geodata_coordinates(df)

    # Select only needed columns
    df = df[["tags.name", "latitude", "longitude"]].rename(
        columns={"tags.name": "name"}
    )

    # Remove rows without required values
    df = df.dropna(subset=["name", "latitude", "longitude"])
    logger.info(f"After removing incomplete records: {len(df)} MRT locations")

    # Handle duplicates by taking mean coordinates
    return handle_geodata_duplicates(df, "MRT")


def clean_mall_geodata(mall_geodata: Any) -> pd.DataFrame:
    """Clean shopping mall geographic data from Overpass API response.

    Similar to MRT geodata cleaning, but for shopping malls. Converts JSON
    response into clean DataFrame and handles duplicates.
    """
    logger.info("Converting mall geodata from JSON...")
    df = convert_overpass_json_to_dataframe(mall_geodata)
    logger.info(f"Converted {len(df)} raw mall geodata records")

    # Process coordinates from different Overpass API formats
    df = process_overpass_geodata_coordinates(df)

    # Select only needed columns
    df = df[["tags.name", "latitude", "longitude"]].rename(
        columns={"tags.name": "name"}
    )

    # Remove rows without required values
    df = df.dropna(subset=["name", "latitude", "longitude"])
    logger.info(f"After removing incomplete records: {len(df)} mall locations")

    # Handle duplicates by taking mean coordinates (e.g. Mustafa Centre appears multiple times)
    return handle_geodata_duplicates(df, "mall")


def clean_hdb_address_geodata(hdb_address_geodata: pd.DataFrame) -> pd.DataFrame:
    """Clean geocoded HDB address data by filtering for quality.

    This function filters the geocoded addresses to keep only high-quality results:
    1. Keeps only 'address' type records (vs. approximate matches)
    2. Filters by confidence threshold (>= 80% confidence)
    3. Removes duplicate addresses (same block + street)

    The 'type' and 'confidence' fields come from the geocoding service that
    converted address strings into latitude/longitude coordinates.
    """
    logger.info(f"Cleaning {len(hdb_address_geodata)} HDB address records")

    df = hdb_address_geodata

    # Filter by type - keep only "address" records (geocoding artifact)
    address_only = df[df["type"] == "address"]
    logger.info(f"After filtering by type='address': {len(address_only)} records")
    df = address_only

    # Filter by confidence threshold (geocoding artifact)
    high_confidence = df[df["confidence"] >= GEOCODING_CONFIDENCE_THRESHOLD]
    logger.info(
        f"After filtering by confidence >= {GEOCODING_CONFIDENCE_THRESHOLD}: {len(high_confidence)} records"
    )
    df = high_confidence

    # Remove duplicates by block and street_name, keeping first occurrence
    unique_addresses = df.drop_duplicates(subset=["block", "street_name"], keep="first")
    duplicates_removed = len(df) - len(unique_addresses)
    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate addresses")
    df = unique_addresses

    logger.info(f"Final HDB address data: {len(df)} unique, high-confidence addresses")
    return df


def _parse_storey_range(storey_range: str) -> tuple[int, int, float]:
    """Parse storey range string like '04 TO 06' into min, max, median."""
    try:
        if pd.isna(storey_range):
            return (0, 0, 0.0)

        # Split on 'TO' and convert to integers
        parts = storey_range.split(" TO ")
        min_storey = int(parts[0])
        max_storey = int(parts[1])
        median_storey = (min_storey + max_storey) / 2

        return (min_storey, max_storey, median_storey)

    except (ValueError, IndexError):
        return (0, 0, 0.0)


def _parse_remaining_lease(lease_str, lease_commence_date, sale_date) -> int:
    """Parse remaining lease into total months. Calculate from sale date and lease commence year if missing."""
    import re

    # Case 1: String format like "56 years 09 months" or "63 years"
    if isinstance(lease_str, str):
        total_months = 0

        # Extract years
        year_match = re.search(r"(\d+)\s*year", lease_str.lower())
        if year_match:
            total_months += int(year_match.group(1)) * 12

        # Extract months
        month_match = re.search(r"(\d+)\s*month", lease_str.lower())
        if month_match:
            total_months += int(month_match.group(1))

        return total_months

    # Case 2: Numeric (older data) - assume it's years
    elif isinstance(lease_str, (int, float)) and not pd.isna(lease_str):
        return int(lease_str * 12)

    # Case 3: Missing - calculate from lease commence date (99-year lease standard)
    elif (
        pd.isna(lease_str)
        and not pd.isna(lease_commence_date)
        and not pd.isna(sale_date)
    ):
        sale_year = sale_date.year
        years_elapsed = sale_year - int(lease_commence_date)
        remaining_years = max(0, 99 - years_elapsed)  # HDB leases are 99 years
        return remaining_years * 12

    return 0


def _extract_room_count(flat_type: str) -> int:
    """Extract room count from flat_type like '3 ROOM' or special cases 'EXECUTIVE'/'MULTI-GENERATION'."""
    try:
        if pd.isna(flat_type):
            return 0

        flat_type = flat_type.upper().strip()

        # Handle special cases
        if "EXECUTIVE" in flat_type or "MULTI-GENERATION" in flat_type:
            return 6

        # Extract number from "3 ROOM" pattern
        import re

        room_match = re.search(r"(\d+)", flat_type)
        if room_match:
            return int(room_match.group(1))

        return 0

    except (ValueError, AttributeError):
        return 0


def _count_mrt_station_lines(station_code: str) -> int:
    """Count number of lines serving a station from codes like 'NS1 EW24'."""
    # Simply count the number of space-separated codes
    return len(station_code.split())
