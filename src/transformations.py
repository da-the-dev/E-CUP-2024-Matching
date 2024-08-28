__all__ = ["categories_transformer", "attr_transformer"]

import json
from sklearn.decomposition import PCA
import torch
import random
import numpy as np
import torch.nn as nn
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
    def __init__(self):
        super().__init__()

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


class bert_64_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, model, feature: str):
        super().__init__()

        self.model = model
        self.feature = feature
        self.pca = PCA(n_components=64)

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # self.projection_layer = nn.Linear(768, 64).to(self.device)

    def transform(self, X):
        # Fix seeds for reproducibility
        def set_seed(seed):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            random.seed(seed)
            torch.backends.cudnn.deterministic = True

        set_seed(42)
        
        # Apply the encoding function to the specified feature
        X[self.feature] = list(self.model.encode(X[self.feature]))

        return X