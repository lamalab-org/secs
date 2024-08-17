import pandas as pd
import polars as pl


def csv_load_function(path: str) -> pl.DataFrame:
    return pl.read_csv(path)


def pickle_load_function(path: str) -> pl.DataFrame:
    data = pd.read_pickle(path)
    return pl.from_pandas(data)


def parquet_load_function(path: str) -> pl.DataFrame:
    data = pd.read_parquet(path)
    return pl.from_pandas(data)