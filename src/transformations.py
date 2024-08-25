__all__ = ["categories_transformer", "attr_transformer"]

import json
from sklearn.base import BaseEstimator, TransformerMixin


class categories_transformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def transform(self, X):
        def unjson(x):
            categories = set(json.loads(x).values())
            if "EPG" in categories:
                categories.remove("EPG")

            return " ".join(categories)

        X["categories"] = X["categories"].transform(unjson)

        return X


class attr_transformer(BaseEstimator, TransformerMixin):
    def init(self, feature: str):
        super().init()
        self.feature = feature

    def transform(self, X):
        def attr_t(x):
            result_str = ""

            row_dct: dict = json.loads(x)

            items = list(row_dct.items())
            items.sort()
            for key, val in items:
                result_str += key + ":"
                val_str = ", ".join(val)
                result_str += val_str
                result_str += ";"
            return result_str

        X["characteristic_attributes_mapping"] = X[
            "characteristic_attributes_mapping"
        ].transform(attr_t)

        return X
