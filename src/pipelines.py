from sklearn.pipeline import Pipeline

from src.transformations import (
    attr_transformer,
    bert_64_transformer,
    categories_transformer,
)


def embedding_pipeline(model, tokenizer):
    return Pipeline(
        [
            ("categories_initial", categories_transformer("categories")),
            ("attr_initial", attr_transformer("characteristic_attributes_mapping")),
            ("categories_embed", bert_64_transformer(model, tokenizer, "categories")),
            (
                "attr_embed",
                bert_64_transformer(
                    model, tokenizer, "characteristic_attributes_mapping"
                ),
            ),
        ]
    )
