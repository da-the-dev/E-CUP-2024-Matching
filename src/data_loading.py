import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame


def load_data_full(path):
    """Read .parquet in full"""
    return pd.read_parquet(path)


def load_data_batch(path, batch_size=10_000):
    """Read a singular batch from .parquet file"""
    parquet_file = pq.ParquetFile(path)

    batch = next(parquet_file.iter_batches(batch_size))
    df: DataFrame = batch.to_pandas()
    return df
