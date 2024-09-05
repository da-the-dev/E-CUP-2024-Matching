__all__ = ["categories_transformer", "attr_transformer"]

import json
import re
from typing_extensions import deprecated
from sklearn.decomposition import PCA
import torch
import random
import numpy as np
import torch.nn as nn
from sklearn.base import BaseEstimator, TransformerMixin


class categories_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        super().__init__()
        self.feature = feature

    def __unjson(self, x):
        categories = set(json.loads(x).values())
        if "EPG" in categories:
            categories.remove("EPG")

        return " ".join(categories)

    def transform(self, X):
        X_copy = X.copy()

        X_copy[self.feature] = X[self.feature].transform(self.__unjson)

        return X_copy


class attr_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature):
        super().__init__()
        self.feature = feature

    def __attr_t(self,  x):
        result_str = ""

        row_dct: dict = json.loads(x)

        items = list(row_dct.items())
        items.sort()
        for key, val in items:
            result_str += key + " "
            val_str = " ".join(val)
            result_str += val_str
            result_str += " "
        result_str = re.sub(r"[^A-zА-я\s\d][\\\^]?", "", result_str)
        result_str = re.sub(r"\s{2,}", " ", result_str)
        return result_str

    def transform(self, X):
        X_copy = X.copy()

        X_copy[self.feature] = X[self.feature].transform(self.__attr_t)

        return X_copy

@deprecated("New transformer is faster")
class old_bert_64_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, model, tokenizer, feature: str):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.feature = feature
        self.pca = PCA(n_components=64)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def transform(self, X):
        self.set_seed(42)

        # List to hold embeddings for PCA fitting
        embeddings = []

        # Function to encode and collect embeddings
        def encode_string(string):
            tokens = self.tokenizer.encode_plus(
                string, return_tensors="pt", max_length=1024, truncation=True
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**tokens)

            attention_mask = tokens["attention_mask"].squeeze()
            token_embeddings = output.last_hidden_state.squeeze()
            token_embeddings = token_embeddings[attention_mask == 1]
            pooled_embedding = (
                torch.mean(token_embeddings, dim=0).detach().cpu().numpy()
            )

            embeddings.append(pooled_embedding)
            return pooled_embedding

        # Apply encoding to each string
        X[self.feature] = X[self.feature].apply(encode_string)

        # Convert collected embeddings to a numpy array for PCA
        embeddings = np.array(embeddings)

        # Fit PCA on collected embeddings and transform them
        reduced_embeddings = self.pca.fit_transform(embeddings)

        # Update DataFrame with reduced embeddings
        X[self.feature] = list(reduced_embeddings)

        return X


class bert_64_transformer(BaseEstimator, TransformerMixin):
    def __init__(self, model, tokenizer, feature: str, batch_size: int = 512):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model
        self.feature = feature
        self.pca = PCA(n_components=64)
        self.batch_size = batch_size

        # Определяем устройство: MPS (GPU на Mac) или CUDA (NVIDIA GPU)
        self.device = (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        )
        self.model.to(self.device)  # Перемещаем модель на GPU

    def __set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def __encode_batch(self, texts):
        # Токенизация батча текстов
        tokens = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,  # Уменьшение max_length для снижения потребления памяти
        ).to(self.device)

        # Получение эмбеддингов с использованием модели
        with torch.no_grad():
            output = self.model(**tokens)

        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        embeddings = output.last_hidden_state * attention_mask

        # Усреднение эмбеддингов по не нулевым токенам
        pooled_embeddings = embeddings.sum(dim=1) / attention_mask.sum(dim=1)

        return pooled_embeddings

    def transform(self, X):
        X_copy = X.copy()
        self.__set_seed(42)

        all_texts = X[self.feature].tolist()
        all_embeddings = []

        for i in range(0, len(all_texts), self.batch_size):
            batch_texts = all_texts[i : i + self.batch_size]
            batch_embeddings = self.__encode_batch(batch_texts)
            all_embeddings.append(batch_embeddings.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        reduced_embeddings = self.pca.fit_transform(all_embeddings)

        X_copy[self.feature] = list(reduced_embeddings)
        return X_copy
