import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame
import transformations as tf
from transformers import BertModel, BertTokenizer


def load_data():
    parquet_file = pq.ParquetFile("data/train/merged_data.parquet")


    batch = next(parquet_file.iter_batches())
    df: DataFrame = batch.to_pandas()
    return df


def make_transformations_1(df):
    new_df = df
    tokenizer = BertTokenizer.from_pretrained('sergeyzh/rubert-tiny-turbo')
    model = BertModel.from_pretrained('sergeyzh/rubert-tiny-turbo')

    tf.attr_transformer("characteristic_attributes_mapping").transform(new_df)
    tf.categories_transformer('categories').transform(new_df)

    tf.bert_64_transformer2(model, tokenizer, 'categories').fit_transform(new_df)
    tf.bert_64_transformer(model, tokenizer, "characteristic_attributes_mapping").transform(new_df)
    return new_df


# df = load_data()
# print(df.head())
# print("\n================================")
# print("================================\n")
# new_df = make_transformations_1(df)
# print(new_df.head())
import polars as pl
print("done")