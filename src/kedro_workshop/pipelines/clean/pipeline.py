"""
This is a boilerplate pipeline 'clean'
generated using Kedro 1.0.0
"""

from kedro.pipeline import Node, Pipeline

from .nodes import (
    clean_hdb_resale_prices,
    clean_mrt_stations,
    clean_mrt_geodata,
    clean_mall_geodata,
    clean_hdb_address_geodata,
)


def create_pipeline(**kwargs) -> Pipeline:
    """This pipeline makes the data consistent and reliable."""
    return Pipeline(
        [
            Node(
                func=clean_hdb_resale_prices,
                inputs="staging_hdb_resale_prices",
                outputs="clean_hdb_resale_prices",
                name="clean_hdb_resale_prices_node",
            ),
            Node(
                func=clean_mrt_stations,
                inputs="staging_mrt_stations",
                outputs="clean_mrt_stations",
                name="clean_mrt_stations_node",
            ),
            Node(
                func=clean_mrt_geodata,
                inputs="staging_mrt_geodata",
                outputs="clean_mrt_geodata",
                name="clean_mrt_geodata_node",
            ),
            Node(
                func=clean_mall_geodata,
                inputs="staging_mall_geodata",
                outputs="clean_mall_geodata",
                name="clean_mall_geodata_node",
            ),
            Node(
                func=clean_hdb_address_geodata,
                inputs="staging_hdb_address_geodata",
                outputs="clean_hdb_address_geodata",
                name="clean_hdb_address_geodata_node",
            ),
        ]
    )
