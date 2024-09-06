import pickle
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from src.transformations import (
    attr_transformer,
    categories_transformer,
    bert_64_transformer,
)


def prepare_data(
    resnet_path,
    text_and_bert_path,
    attributes_path,
    data_path,
    tokenizer,
    embedder,
    catboost=False,
    train=False,
    cache_path=None,
):
    if cache_path is None:
        print("WARNING! DATA CACHING IS DISABLED")
    else:
        print("Cache path defined, loading data from cache..")
        try:
            return pd.read_parquet(cache_path, engine="pyarrow")
        except FileNotFoundError:
            print("Cache does not exist. Will write data to cache_path")
            

    # Read parquets
    print("Reading parquets...")
    resnet = pd.read_parquet(
        resnet_path,
        engine="pyarrow",
        columns=["variantid", "main_pic_embeddings_resnet_v1"],
    )
    text_and_bert = pd.read_parquet(
        text_and_bert_path,
        engine="pyarrow",
        columns=["variantid", "name_bert_64"],
    )
    attributes = pd.read_parquet(
        attributes_path,
        engine="pyarrow",
    )
    data = pd.read_parquet(
        data_path,
        engine="pyarrow",
    )

    # Merging
    print("Merging...")
    # Merge resnet
    full_data = data.merge(
        resnet, left_on="variantid1", right_on="variantid", how="left"
    )
    full_data = full_data.rename(
        columns={"main_pic_embeddings_resnet_v1": "pic_embeddings_1"}
    )
    full_data = full_data.drop(columns=["variantid"])

    full_data = full_data.merge(
        resnet, left_on="variantid2", right_on="variantid", how="left"
    )
    full_data = full_data.rename(
        columns={"main_pic_embeddings_resnet_v1": "pic_embeddings_2"}
    )
    full_data = full_data.drop(columns=["variantid"])
    print("     resnet..")

    # Merge text_and_bert
    full_data = full_data.merge(
        text_and_bert, left_on="variantid1", right_on="variantid", how="left"
    )
    full_data = full_data.rename(columns={"name_bert_64": "name_bert_64_1"})
    full_data = full_data.drop(columns=["variantid"])

    full_data = full_data.merge(
        text_and_bert, left_on="variantid2", right_on="variantid", how="left"
    )
    full_data = full_data.rename(columns={"name_bert_64": "name_bert_64_2"})
    full_data = full_data.drop(columns=["variantid"])
    print("     bert...")

    # Merge attributes
    full_data = full_data.merge(
        attributes, left_on="variantid1", right_on="variantid", how="left"
    )
    full_data = full_data.rename(
        columns={
            "characteristic_attributes_mapping": "attributes_1",
            "categories": "categories_1",
        }
    )
    full_data = full_data.drop(columns=["variantid"])

    full_data = full_data.merge(
        attributes, left_on="variantid2", right_on="variantid", how="left"
    )
    full_data = full_data.rename(
        columns={
            "characteristic_attributes_mapping": "attributes_2",
            "categories": "categories_2",
        }
    )
    full_data = full_data.drop(columns=["variantid"])
    print("     attributes and categories...")

    # Transform columns
    print("Transforming columns...")
    attr_transformer("attributes_1").transform(full_data)
    print("     attributes_1...")
    categories_transformer("categories_1").transform(full_data)
    print("     categories_1...")
    attr_transformer("attributes_2").transform(full_data)
    print("     attributes_2...")
    categories_transformer("categories_2").transform(full_data)
    print("     categories_2...")

    # Embed
    print("Embedding...")
    print("     attributes_1...")
    full_data = bert_64_transformer(embedder, tokenizer, "attributes_1").transform(
        full_data
    )
    print("     categories_1...")
    full_data = bert_64_transformer(embedder, tokenizer, "categories_1").transform(
        full_data
    )
    print("     attributes_2...")
    full_data = bert_64_transformer(embedder, tokenizer, "attributes_2").transform(
        full_data
    )
    print("     categories_2...")
    full_data = bert_64_transformer(embedder, tokenizer, "categories_2").transform(
        full_data
    )

    # Concat embeddings
    print("Concating embeddings...")
    full_data["concated_embeddings_1"] = full_data.apply(
        lambda row: np.concatenate(
            (
                row["pic_embeddings_1"][0],
                row["name_bert_64_1"],
                row["categories_1"],
                row["attributes_1"],
            )
        ),
        axis=1,
    )
    full_data["concated_embeddings_2"] = full_data.apply(
        lambda row: np.concatenate(
            (
                row["pic_embeddings_2"][0],
                row["name_bert_64_2"],
                row["categories_2"],
                row["attributes_1"],
            )
        ),
        axis=1,
    )

    print("Final preparation...")
    if train:
        full_data = full_data[
            [
                "variantid1",
                "variantid2",
                "concated_embeddings_1",
                "concated_embeddings_2",
                "target",
            ]
        ]
    else:
        full_data = full_data[
            [
                "variantid1",
                "variantid2",
                "concated_embeddings_1",
                "concated_embeddings_2",
            ]
        ]

    # CAT BOOST SPECIFIC
    if catboost:
        print("Running catboost-specific spreading...")

        print("     concated_embeddings_2...")
        expanded_columns = pd.DataFrame(
            full_data["concated_embeddings_2"].tolist(), index=full_data.index
        )
        full_data = pd.concat(
            [full_data.drop("concated_embeddings_2", axis=1), expanded_columns], axis=1
        )

        # Rename columns
        new_columns = {i: i + 320 for i in range(320)}
        full_data.rename(columns=new_columns, inplace=True)

        # Spread embedding1
        print("     concated_embeddings_1...")
        expanded_columns = pd.DataFrame(
            full_data["concated_embeddings_1"].tolist(), index=full_data.index
        )
        full_data = pd.concat(
            [full_data.drop("concated_embeddings_1", axis=1), expanded_columns], axis=1
        )

    if cache_path:
        print(f"Writing resulting data to {cache_path}...")
        full_data.to_parquet(cache_path, engine="pyarrow")

    print("\nDone!")

    return full_data
