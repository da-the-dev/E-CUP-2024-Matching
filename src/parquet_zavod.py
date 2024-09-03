from typing import Callable
from pandas import DataFrame
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

def parquet_zavod(
    input: str,
    output: str,
    output_schema,
    func: Callable[[DataFrame], DataFrame],
    batch_size=64000,
):
    # Open the input Parquet file
    input_file = pq.ParquetFile(input)

    # Setup progress tracking
    total_rows = input_file.metadata.num_rows

    # Open the output Parquet file for writing
    writer = pq.ParquetWriter(output, output_schema)

    # Process the file in batches
    with tqdm(total=total_rows, desc="Processing", unit="rows") as pbar:
        for batch in input_file.iter_batches(batch_size):
            # Convert the batch to a pandas DataFrame for processing
            df = batch.to_pandas()
            
            df = func(df)
            # Update progress
            pbar.update(len(df))

            # Convert the processed DataFrame back to a PyArrow Table
            processed_table = pa.Table.from_pandas(df)

            # Write the processed batch to the output file
            writer.write_table(processed_table)

        # Close the writer
        writer.close()
