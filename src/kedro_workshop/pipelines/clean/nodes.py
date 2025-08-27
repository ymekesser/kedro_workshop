import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Constants for data cleaning
GEOCODING_CONFIDENCE_THRESHOLD = 0.8


def clean_hdb_resale_prices(hdb_resale_prices: pd.DataFrame) -> pd.DataFrame:
    """Clean HDB resale prices data with feature parsing and data type conversions."""
    logger.info(f"Cleaning {len(hdb_resale_prices)} HDB resale price records")
    
    # Work directly with the data - Kedro handles data isolation
    df = hdb_resale_prices
    
    # Remove any unnamed/extra columns (like 'Column1')
    df = df.loc[:, ~df.columns.str.contains(r'^Unnamed|Column\d+$', case=False, na=False)]
    
    # Convert month to datetime
    if 'month' in df.columns:
        df['month'] = pd.to_datetime(df['month'], format='%m/%d/%Y', errors='coerce')
        logger.info(f"Date range: {df['month'].min()} to {df['month'].max()}")
    
    # Convert basic numeric columns
    numeric_columns = ['resale_price', 'floor_area_sqm', 'lease_commence_date']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Parse storey range to get min, max, and median storey
    if 'storey_range' in df.columns:
        logger.info("Parsing storey ranges...")
        df[['storey_min', 'storey_max', 'storey_median']] = df['storey_range'].apply(
            lambda x: pd.Series(_parse_storey_range(x))
        )
    
    # Parse remaining lease to convert to months
    if 'remaining_lease' in df.columns:
        logger.info("Parsing remaining lease...")
        df['remaining_lease_months'] = df.apply(
            lambda row: _parse_remaining_lease(
                row['remaining_lease'], 
                row.get('lease_commence_date'), 
                row.get('month')
            ), 
            axis=1
        )
    
    # Extract room count from flat_type
    if 'flat_type' in df.columns:
        logger.info("Extracting room count...")
        df['room_count'] = df['flat_type'].apply(_extract_room_count)
    
    logger.info(f"Cleaned data: {len(df)} records")
    logger.info(f"Added derived columns: storey_min, storey_max, storey_median, remaining_lease_months, room_count")
    return df


def clean_mrt_stations(mrt_stations: pd.DataFrame) -> pd.DataFrame:
    """Clean MRT stations data - filter operational stations, parse dates, count lines."""
    logger.info(f"Cleaning {len(mrt_stations)} MRT station records")
    
    df = mrt_stations
    
    # Convert Opening column to datetime, filter out planned stations
    logger.info("Converting opening dates and filtering operational stations...")
    df['opening_date'] = pd.to_datetime(df['Opening'], format='%m/%d/%Y', errors='coerce')
    
    # Remove stations where opening date couldn't be parsed (planned stations)
    operational_stations = df.dropna(subset=['opening_date'])
    logger.info(f"Filtered out {len(df) - len(operational_stations)} planned stations")
    df = operational_stations
    
    # Count number of lines per station from station codes
    logger.info("Counting lines per station...")
    df['line_count'] = df['Code'].apply(_count_station_lines)
    
    logger.info(f"Cleaned data: {len(df)} operational stations")
    return df


def clean_mrt_geodata(mrt_geodata: Any) -> pd.DataFrame:
    """Clean MRT geodata - convert JSON, select needed columns, handle duplicates."""
    logger.info("Converting MRT geodata from JSON...")
    df = _convert_overpass_json_to_dataframe(mrt_geodata)
    logger.info(f"Converted {len(df)} raw MRT geodata records")
    
    # Coalesce lat/center.lat and lon/center.lon, rename to latitude/longitude
    center_lat = df.get('center.lat', pd.Series(index=df.index, dtype='float64'))
    center_lon = df.get('center.lon', pd.Series(index=df.index, dtype='float64'))
    df['latitude'] = df['lat'].fillna(center_lat)
    df['longitude'] = df['lon'].fillna(center_lon)
    
    # Select only needed columns
    df = df[['tags.name', 'latitude', 'longitude']].rename(columns={'tags.name': 'name'})
    
    # Remove rows without required values
    df = df.dropna(subset=['name', 'latitude', 'longitude'])
    logger.info(f"After removing incomplete records: {len(df)} MRT locations")
    
    # Handle duplicates by taking mean coordinates
    return _handle_geodata_duplicates(df, "MRT")


def clean_mall_geodata(mall_geodata: Any) -> pd.DataFrame:
    """Clean mall geodata - convert JSON, select needed columns, handle duplicates."""
    logger.info("Converting mall geodata from JSON...")
    df = _convert_overpass_json_to_dataframe(mall_geodata)
    logger.info(f"Converted {len(df)} raw mall geodata records")
    
    # Coalesce lat/center.lat and lon/center.lon, rename to latitude/longitude
    center_lat = df.get('center.lat', pd.Series(index=df.index, dtype='float64'))
    center_lon = df.get('center.lon', pd.Series(index=df.index, dtype='float64'))
    df['latitude'] = df['lat'].fillna(center_lat)
    df['longitude'] = df['lon'].fillna(center_lon)
    
    # Select only needed columns
    df = df[['tags.name', 'latitude', 'longitude']].rename(columns={'tags.name': 'name'})
    
    # Remove rows without required values
    df = df.dropna(subset=['name', 'latitude', 'longitude'])
    logger.info(f"After removing incomplete records: {len(df)} mall locations")
    
    # Handle duplicates by taking mean coordinates (e.g. Mustafa Centre appears multiple times)
    return _handle_geodata_duplicates(df, "mall")


def clean_hdb_address_geodata(hdb_address_geodata: pd.DataFrame) -> pd.DataFrame:
    """Clean HDB address geodata - filter by confidence, type, and remove duplicates.
    
    Note: 'type' and 'confidence' are artifacts from the geocoding service used to 
    generate latitude/longitude coordinates from address strings.
    """
    logger.info(f"Cleaning {len(hdb_address_geodata)} HDB address records")
    
    df = hdb_address_geodata
    
    # Filter by type - keep only "address" records (geocoding artifact)
    if 'type' in df.columns:
        address_only = df[df['type'] == 'address']
        logger.info(f"After filtering by type='address': {len(address_only)} records")
        df = address_only
    
    # Filter by confidence threshold (geocoding artifact)
    if 'confidence' in df.columns:
        high_confidence = df[df['confidence'] >= GEOCODING_CONFIDENCE_THRESHOLD]
        logger.info(f"After filtering by confidence >= {GEOCODING_CONFIDENCE_THRESHOLD}: {len(high_confidence)} records")
        df = high_confidence
    
    # Remove duplicates by block and street_name, keeping first occurrence
    if 'block' in df.columns and 'street_name' in df.columns:
        unique_addresses = df.drop_duplicates(subset=['block', 'street_name'], keep='first')
        duplicates_removed = len(df) - len(unique_addresses)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate addresses")
        df = unique_addresses
    
    logger.info(f"Final HDB address data: {len(df)} unique, high-confidence addresses")
    return df


def _convert_overpass_json_to_dataframe(data: Any):
    data_dict = pd.json_normalize(data, record_path=["elements"])
    return pd.DataFrame.from_dict(data_dict, orient="columns")  # type: ignore


def _parse_storey_range(storey_range: str) -> tuple[int, int, float]:
    """Parse storey range string like '04 TO 06' into min, max, median."""
    try:
        if pd.isna(storey_range):
            return (0, 0, 0.0)
        
        # Split on 'TO' and convert to integers
        parts = storey_range.split(' TO ')
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
        year_match = re.search(r'(\d+)\s*year', lease_str.lower())
        if year_match:
            total_months += int(year_match.group(1)) * 12
        
        # Extract months
        month_match = re.search(r'(\d+)\s*month', lease_str.lower())
        if month_match:
            total_months += int(month_match.group(1))
        
        return total_months
    
    # Case 2: Numeric (older data) - assume it's years
    elif isinstance(lease_str, (int, float)) and not pd.isna(lease_str):
        return int(lease_str * 12)
    
    # Case 3: Missing - calculate from lease commence date (99-year lease standard)
    elif pd.isna(lease_str) and not pd.isna(lease_commence_date) and not pd.isna(sale_date):
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
        if 'EXECUTIVE' in flat_type or 'MULTI-GENERATION' in flat_type:
            return 6
        
        # Extract number from "3 ROOM" pattern
        import re
        room_match = re.search(r'(\d+)', flat_type)
        if room_match:
            return int(room_match.group(1))
        
        return 0
        
    except (ValueError, AttributeError):
        return 0


def _count_station_lines(station_code: str) -> int:
    """Count number of lines serving a station from codes like 'NS1 EW24'."""
    # Simply count the number of space-separated codes
    return len(station_code.split())


def _handle_geodata_duplicates(df: pd.DataFrame, location_type: str) -> pd.DataFrame:
    """Handle duplicates by taking mean coordinates (assuming duplicates occur clustered closely together)."""
    if len(df) == 0:
        return df
    
    cleaned_df = df.groupby('name').agg({
        'latitude': 'mean',
        'longitude': 'mean'
    }).reset_index()
    
    duplicates_removed = len(df) - len(cleaned_df)
    if duplicates_removed > 0:
        logger.info(f"Merged {duplicates_removed} duplicate {location_type} locations")
    
    logger.info(f"Final {location_type} geodata: {len(cleaned_df)} unique locations")
    return cleaned_df
