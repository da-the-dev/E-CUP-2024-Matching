import pyarrow.parquet as pq
import pandas as pd
import yaml


def read_parquet_in_batches(file_path: str, batch_size=65536, load_percentage=100, return_batch=False) -> pd.DataFrame:
    """
    Reads a Parquet file in batches and processes each batch.

    Parameters:
    - file_path (str): The path to the Parquet file.
    - batch_size (int): The number of rows to include in each batch.
    - load_percentage (int): The percentage of data to load from the source. Default is 100%.

    Returns:
    - pd.DataFrame: A concatenated Pandas DataFrame of all batches.
    """

    parquet_file = pq.ParquetFile(file_path)

    total_rows = parquet_file.metadata.num_rows
    rows_to_load = int(total_rows * (load_percentage / 100))
    processed_rows = 0
    batches = []

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        batch_df = batch.to_pandas()

        processed_rows += len(batch_df)
        progress = (processed_rows / rows_to_load) * 100
        print(f'Progress: {progress:.2f}%')
        
        if return_batch:
            yield batch_df

        batches.append(batch_df)

        if processed_rows >= rows_to_load:
            break

    return pd.concat(batches, ignore_index=True)


def config():
    with open('../params.yaml', 'r') as file:
        return yaml.safe_load(file)


def generate_merge_table() -> pd.DataFrame:
    # Load configuration file
    params = config()

    # Define file paths
    attributes_path = '../' + params['attributes_path']
    resnet_path = '../' + params['resnet_path']
    text_and_bert_path = '../' + params['text_and_bert_path']

    # Read the Parquet files in batches
    df_attributes = read_parquet_in_batches(attributes_path)
    df_resnet = read_parquet_in_batches(resnet_path)
    df_text_and_bert = read_parquet_in_batches(text_and_bert_path)

    # Merge the DataFrames on 'variantid'
    merged_df = df_attributes.merge(df_resnet, on='variantid', how='outer')
    merged_df = merged_df.merge(df_text_and_bert, on='variantid', how='outer')

    output_path = '../' + params['merged_df_path']
    merged_df.to_parquet(output_path, index=False)

    return merged_df


if __name__ == '__main__':
    df = generate_merge_table()
    print(df.head())
