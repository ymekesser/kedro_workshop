"""
Transform pipeline for creating ML feature set with HDB resale prices and distance features.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth in kilometers."""
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Radius of Earth in kilometers
    r = 6371
    return c * r


def _find_nearest_locations(
    hdb_addresses: pd.DataFrame, 
    locations: pd.DataFrame,
    location_type: str
) -> pd.DataFrame:
    """Find nearest location (MRT/Mall) for each HDB address."""
    logger.info(f"Finding nearest {location_type} for {len(hdb_addresses)} HDB addresses")
    
    distances = []
    
    for _, hdb in hdb_addresses.iterrows():
        hdb_lat, hdb_lon = hdb['latitude'], hdb['longitude']
        
        # Calculate distances to all locations
        location_distances = []
        for _, location in locations.iterrows():
            distance = _calculate_distance_km(
                hdb_lat, hdb_lon, 
                location['latitude'], location['longitude']
            )
            location_distances.append({
                'name': location['name'],
                'distance_km': distance
            })
        
        # Find nearest
        nearest = min(location_distances, key=lambda x: x['distance_km'])
        distances.append({
            'block': hdb['block'],
            'street_name': hdb['street_name'],
            f'nearest_{location_type}_name': nearest['name'],
            f'nearest_{location_type}_distance_km': nearest['distance_km']
        })
    
    result_df = pd.DataFrame(distances)
    logger.info(f"Average nearest {location_type} distance: {result_df[f'nearest_{location_type}_distance_km'].mean():.2f} km")
    
    return result_df


def create_feature_set(
    hdb_resale_prices: pd.DataFrame,
    hdb_addresses: pd.DataFrame,
    mrt_stations: pd.DataFrame,
    mrt_geodata: pd.DataFrame,
    mall_geodata: pd.DataFrame
) -> pd.DataFrame:
    """Create ML feature set by joining HDB data with distance features."""
    logger.info("Creating ML feature set with distance features")
    
    # Filter MRT geodata to only include operational stations
    operational_mrt = mrt_geodata[mrt_geodata['name'].isin(mrt_stations['Name'])]
    logger.info(f"Filtered to {len(operational_mrt)} operational MRT stations (from {len(mrt_geodata)} total)")
    
    # Find nearest MRT and Mall for each address
    mrt_distances = _find_nearest_locations(hdb_addresses, operational_mrt, "mrt")
    mall_distances = _find_nearest_locations(hdb_addresses, mall_geodata, "mall")
    
    # Merge distance features
    address_features = mrt_distances.merge(
        mall_distances, 
        on=['block', 'street_name'], 
        how='inner'
    )
    
    # Join with HDB resale prices
    feature_set = hdb_resale_prices.merge(
        address_features,
        on=['block', 'street_name'],
        how='inner'
    )
    
    # Select ML-ready features
    ml_features = [
        'resale_price',  # target
        'floor_area_sqm', 
        'room_count', 
        'remaining_lease_months',
        'storey_median',
        'nearest_mrt_distance_km',
        'nearest_mall_distance_km'
    ]
    
    # Keep only complete cases
    feature_set = feature_set[ml_features].dropna()
    
    logger.info(f"Final feature set: {len(feature_set)} records with {len(ml_features)} features")
    logger.info(f"Features: {ml_features}")
    
    return feature_set

