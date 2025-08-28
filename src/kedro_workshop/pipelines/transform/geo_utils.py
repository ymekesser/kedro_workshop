"""Geographic utility functions for distance calculations and location analysis.

Contains reusable functions for working with geographic data, specifically
distance calculations and nearest neighbor analysis.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Earth's radius in kilometers (for Haversine formula)
EARTH_RADIUS_KM = 6371


def calculate_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points on Earth.

    Uses the Haversine formula to calculate the shortest distance between
    two points on the surface of a sphere (Earth).

    Args:
        lat1, lon1: Latitude and longitude of first point in decimal degrees
        lat2, lon2: Latitude and longitude of second point in decimal degrees

    Returns:
        Distance between points in kilometers

    Example:
        >>> # Distance between Singapore landmarks
        >>> calculate_distance_km(1.2858, 103.8494, 1.2966, 103.8520)
        1.23
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    return c * EARTH_RADIUS_KM


def find_nearest_locations(
    source_locations: pd.DataFrame, target_locations: pd.DataFrame, location_type: str
) -> pd.DataFrame:
    """Find the nearest target location for each source location.

    For each location in source_locations, finds the closest location
    in target_locations using great circle distance.

    Args:
        source_locations: DataFrame with 'latitude', 'longitude' columns
        target_locations: DataFrame with 'name', 'latitude', 'longitude' columns
        location_type: Type of target locations for logging (e.g., 'mrt', 'mall')

    Returns:
        DataFrame with nearest location info for each source location.
        Includes original source columns plus:
        - 'nearest_{location_type}_name': Name of nearest location
        - 'nearest_{location_type}_distance_km': Distance to nearest location

    Note:
        This function assumes source_locations has 'block' and 'street_name'
        columns for HDB addresses. For other use cases, modify accordingly.
        This is a naive implementatoin and could be optimized with a KDTree.
    """
    logger.info(
        f"Finding nearest {location_type} for {len(source_locations)} locations"
    )

    distances = []

    for _, source in source_locations.iterrows():
        source_lat, source_lon = source["latitude"], source["longitude"]

        # Calculate distances to all target locations
        location_distances = []
        for _, target in target_locations.iterrows():
            distance = calculate_distance_km(
                source_lat, source_lon, target["latitude"], target["longitude"]
            )
            location_distances.append({"name": target["name"], "distance_km": distance})

        # Find nearest location
        nearest = min(location_distances, key=lambda x: x["distance_km"])
        distances.append(
            {
                "block": source["block"],
                "street_name": source["street_name"],
                f"nearest_{location_type}_name": nearest["name"],
                f"nearest_{location_type}_distance_km": nearest["distance_km"],
            }
        )

    result_df = pd.DataFrame(distances)
    avg_distance = result_df[f"nearest_{location_type}_distance_km"].mean()
    logger.info(f"Average nearest {location_type} distance: {avg_distance:.2f} km")

    return result_df
