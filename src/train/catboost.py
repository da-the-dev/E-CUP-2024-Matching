import os
import pickle

from catboost import CatBoostClassifier
import pandas as pd
from src.data_preparation.prepare_data import prepare_data
from sklearn.model_selection import train_test_split

# Prep file paths
resnet_path = "data/train/resnet.parquet"
text_and_bert_path = "data/train/text_and_bert.parquet"
attributes_path = "data/train/attributes.parquet"
test_path = "data/train/train.parquet"
model_path = "models/model_catboost_2.pkl"
train_data_path = "data/train/train_data.parquet"
train_data_catboost_path = "data/train/train_data_catboost.parquet"

# Load embedder and tokenizer
tokenizer = pickle.loads(open("tokenizer.pkl", "rb").read())
embedder = pickle.loads(open("embedder.pkl", "rb").read())

# Prep data
train_data = prepare_data(
    resnet_path,
    text_and_bert_path,
    attributes_path,
    test_path,
    tokenizer,
    embedder,
    catboost=True,
    train=True,
    cache_path=train_data_catboost_path,
)

# Split
X = train_data.drop(["target"], axis=1)
y = train_data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Load the model
model = CatBoostClassifier(learning_rate=1, depth=6)

# Fit
model = model.fit(X, y)

# Save weights
with open(model_path, "wb") as file:
    file.write(pickle.dumps(model))

# Test the model
from sklearn.metrics import f1_score, accuracy_score

y_pred = model.predict(X_test)

print("F1:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Saving the test sample for testing
test_data = pd.DataFrame(X_test, columns=train_data.drop(["target"], axis=1).columns)
test_data["target"] = y_test

test_data.to_parquet("data/test/test_sample.parquet", engine="pyarrow")
