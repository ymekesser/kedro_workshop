"""
Transform pipeline for creating ML feature set with HDB resale prices and distance features.
"""

from kedro.pipeline import Node, Pipeline

from .nodes import create_feature_set


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=create_feature_set,
            inputs=[
                "clean_hdb_resale_prices",
                "clean_hdb_address_geodata", 
                "clean_mrt_stations",
                "clean_mrt_geodata",
                "clean_mall_geodata"
            ],
            outputs="transformed_feature_set",
            name="create_feature_set_node",
        )
    ])
