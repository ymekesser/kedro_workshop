"""Transform pipeline for creating ML-ready feature sets.

This pipeline combines cleaned data from multiple sources to create
feature sets suitable for machine learning models. The main transformation
is joining the different datasets and adding geographic proximity features.
"""

from kedro.pipeline import Node, Pipeline

from .nodes import create_feature_set


def create_pipeline(**kwargs) -> Pipeline:
    """Create the data transformation pipeline.

    This pipeline takes cleaned data from multiple sources and creates
    ML-ready feature sets with geographic proximity features.

    The main transformation calculates distance-based features by finding
    the nearest MRT station and shopping mall for each HDB address.
    """
    return Pipeline(
        [
            Node(
                func=create_feature_set,
                inputs=[
                    "clean_hdb_resale_prices",
                    "clean_hdb_address_geodata",
                    "clean_mrt_stations",
                    "clean_mrt_geodata",
                    "clean_mall_geodata",
                ],
                outputs="transformed_feature_set",
                name="create_feature_set_node",
            ),
        ]
    )
