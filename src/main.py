import pyarrow.parquet as pq


def read_parquet_in_batches(file_path, batch_size=65536):
    """
    Reads a Parquet file in batches and processes each batch.

    Parameters:
    - file_path (str): The path to the Parquet file.
    - batch_size (int): The number of rows to include in each batch.

    Yields:
    - pd.DataFrame: A Pandas DataFrame for each batch.
    """
    
    parquet_file = pq.ParquetFile(file_path)
    
    total_rows = parquet_file.metadata.num_rows
    processed_rows = 0
    
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_df = batch.to_pandas()
        
        processed_rows += len(batch_df)
        progress = (processed_rows / total_rows) * 100
        print(f'Progress: {progress:.2f}%')
        
        yield batch_df