import pickle as pkl  # noqa: I002

import polars as pl


def csv_load_function(path: str) -> pl.DataFrame:
    return pl.read_csv(path)

def pickle_load_function(path: str) -> pl.DataFrame:
    data = pkl.load(open(path, "rb"))  # noqa: PTH123, SIM115
    return pl.from_pandas(data)
