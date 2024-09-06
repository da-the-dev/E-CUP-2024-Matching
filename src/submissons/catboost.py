import pickle

import pandas as pd
from src.data_preparation.prepare_data import prepare_data

# Prep file paths
resnet_path = "data/test/resnet_test.parquet"
text_and_bert_path = "data/test/text_and_bert_test.parquet"
attributes_path = "data/test/attributes_test.parquet"
test_path = "data/test/test.parquet"
model_path = "models/model_catboost.pkl"
test_data_catboost_path = "data/test/test_data_catboost.parquet"

# Load embedder and tokenizer
tokenizer = pickle.loads(open("tokenizer.pkl", "rb").read())
embedder = pickle.loads(open("embedder.pkl", "rb").read())

# Prep data
X_test = prepare_data(
    resnet_path,
    text_and_bert_path,
    attributes_path,
    test_path,
    tokenizer,
    embedder,
    catboost=True,
    train=False,
    cache_path=test_data_catboost_path,
)

# Load the model
print("Loading the model...")
model = pickle.loads(open(model_path, "rb").read())

# Run model on the test data
print("Running model on the test data...")
y_pred = model.predict(X_test)

# Saving submisson.csv
test_data = pd.read_parquet(test_path, engine="pyarrow")
test_data["target"] = y_pred

print("Saving submisson.csv...")
test_data.to_parquet("data/submisson.csv")
