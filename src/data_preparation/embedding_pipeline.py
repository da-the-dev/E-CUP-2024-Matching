from sklearn.pipeline import Pipeline

from src.transformations import (
    attr_transformer,
    bert_64_transformer,
    categories_transformer,
)


def embedding_pipeline(embedder, tokenizer, features: tuple[str]):
    return Pipeline(
        [
            (
                "categories_initial",
                categories_transformer(features[0]),
            ),
            (
                "attr_initial",
                attr_transformer(features[1]),
            ),
            (
                "categories_embed",
                bert_64_transformer(embedder, tokenizer, features[0]),
            ),
            (
                "attr_embed",
                bert_64_transformer(embedder, tokenizer, features[1]),
            ),
        ]
    )
