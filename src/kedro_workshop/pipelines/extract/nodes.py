from typing import Any

import pandas as pd
import requests


def extract_hdb_resale_prices(hdb_resale_prices: pd.DataFrame) -> pd.DataFrame:
    return hdb_resale_prices


def extract_mrt_stations(mrt_stations: pd.DataFrame) -> pd.DataFrame:
    return mrt_stations


def extract_mrt_geodata(mrt_geodata: requests.Response) -> Any:
    return mrt_geodata.json()


def extract_mall_geodata(mall_geodata: requests.Response) -> Any:
    return mall_geodata.json()


def extract_hdb_address_geodata(hdb_address_geodata: pd.DataFrame) -> pd.DataFrame:
    return hdb_address_geodata
