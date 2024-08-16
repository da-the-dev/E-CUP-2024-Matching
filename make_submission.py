import pandas as pd
import numpy as np
import joblib
from baseline import process_text_and_bert, merge_data

def load_test_data():
    attributes_path = './data/test/attributes_test.parquet'
    resnet_path = './data/test/resnet_test.parquet'
    text_and_bert_path = './data/test/text_and_bert_test.parquet'
    val_path = './data/test/test.parquet'

    attributes = pd.read_parquet(attributes_path, engine='pyarrow')
    resnet = pd.read_parquet(resnet_path, engine='pyarrow')
    text_and_bert = pd.read_parquet(text_and_bert_path, engine='pyarrow')
    test = pd.read_parquet(val_path, engine='pyarrow')
    
    return attributes, resnet, text_and_bert, test

def prepare_test_data(test_data, tfidf_vectorizer):
    text_data = test_data['text_1'] + ' ' + test_data['text_2']
    text_embeddings = tfidf_vectorizer.transform(text_data).toarray()

    test_data['combined_embeddings'] = test_data.apply(lambda row: np.concatenate([
        row['pic_embeddings_1'][0], row['pic_embeddings_2'][0], text_embeddings[row.name]
    ]), axis=1)

    X_test = np.vstack(test_data['combined_embeddings'].values)

    return X_test

def main():
    _, resnet, text_and_bert, test = load_test_data()
    text_and_bert = process_text_and_bert(text_and_bert)

    test_data = merge_data(test, resnet, text_and_bert)

    tfidf_vectorizer = joblib.load('vectorizer.pkl')

    X_test = prepare_test_data(test_data, tfidf_vectorizer)
    
    model = joblib.load('baseline.pkl')
    
    predictions_prob = model.predict_proba(X_test)[:, 1]
    predictions = (predictions_prob >= 0.5).astype(int)

    submission = pd.DataFrame({
        'variantid1': test['variantid1'],
        'variantid2': test['variantid2'],
        'target': predictions
    })
    
    submission.to_csv('./data/submission.csv', index=False)

if __name__ == "__main__":
    main()
