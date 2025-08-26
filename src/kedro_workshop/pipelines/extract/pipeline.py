from kedro.pipeline import Node, Pipeline

from .nodes import (
    extract_hdb_address_geodata,
    extract_hdb_resale_prices,
    extract_mall_geodata,
    extract_mrt_geodata,
    extract_mrt_stations,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=extract_hdb_resale_prices,
                inputs="source_hdb_resale_prices",
                outputs="staging_hdb_resale_prices",
                name="extract_hdb_resale_prices_node",
            ),
            Node(
                func=extract_mrt_stations,
                inputs="source_mrt_stations",
                outputs="staging_mrt_stations",
                name="extract_mrt_stations_node",
            ),
            Node(
                func=extract_mrt_geodata,
                inputs="source_mrt_geodata",
                outputs="staging_mrt_geodata",
                name="extract_mrt_geodata_node",
            ),
            Node(
                func=extract_mall_geodata,
                inputs="source_mall_geodata",
                outputs="staging_mall_geodata",
                name="extract_mall_geodata_node",
            ),
            Node(
                func=extract_hdb_address_geodata,
                inputs="source_hdb_address_geodata",
                outputs="staging_hdb_address_geodata",
                name="extract_hdb_address_geodata_node",
            ),
        ]
    )
