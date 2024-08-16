import pandas as pd
import numpy as np
import joblib
import json

def load_test_data():
    attributes_path = './data/test/attributes_test.parquet'
    resnet_path = './data/test/resnet_test.parquet'
    text_and_bert_path = './data/test/text_and_bert_test.parquet'
    val_path = './data/test/test.parquet'

    attributes_path = '/Users/nikitakamenev/Documents/repo/hack_ozon/e_cup/data/test_data_task_01/attributes_test.parquet'
    resnet_path = '/Users/nikitakamenev/Documents/repo/hack_ozon/e_cup/data/test_data_task_01/resnet_test.parquet'
    text_and_bert_path = '/Users/nikitakamenev/Documents/repo/hack_ozon/e_cup/data/test_data_task_01/text_and_bert_test.parquet'
    val_path = '/Users/nikitakamenev/Documents/repo/hack_ozon/e_cup/data/test_data_task_01/test.parquet'


    attributes = pd.read_parquet(attributes_path, engine='pyarrow')
    resnet = pd.read_parquet(resnet_path, engine='pyarrow')
    text_and_bert = pd.read_parquet(text_and_bert_path, engine='pyarrow')
    test = pd.read_parquet(val_path, engine='pyarrow')
    
    return attributes, resnet, text_and_bert, test

def process_text_and_bert(df):
    df['categories'] = df['categories'].apply(json.loads)
    df['characteristic_attributes_mapping'] = df['characteristic_attributes_mapping'].apply(json.loads)
    df['combined_text'] = df.apply(lambda row: ' '.join(
        [' '.join(map(str, v)) if isinstance(v, list) else str(v) for v in list(row['categories'].values())] + 
        [' '.join(map(str, v)) if isinstance(v, list) else str(v) for v in list(row['characteristic_attributes_mapping'].values())]
    ), axis=1)
    return df

def merge_data(test, resnet, text_and_bert):
    test_data = test.merge(resnet[['variantid', 'main_pic_embeddings_resnet_v1']], left_on='variantid1', right_on='variantid', how='left')
    test_data = test_data.rename(columns={'main_pic_embeddings_resnet_v1': 'pic_embeddings_1'})
    test_data = test_data.drop(columns=['variantid'])

    test_data = test_data.merge(resnet[['variantid', 'main_pic_embeddings_resnet_v1']], left_on='variantid2', right_on='variantid', how='left')
    test_data = test_data.rename(columns={'main_pic_embeddings_resnet_v1': 'pic_embeddings_2'})
    test_data = test_data.drop(columns=['variantid'])

    test_data = test_data.merge(text_and_bert[['variantid', 'combined_text']], left_on='variantid1', right_on='variantid', how='left')
    test_data = test_data.rename(columns={'combined_text': 'text_1'})
    test_data = test_data.drop(columns=['variantid'])

    test_data = test_data.merge(text_and_bert[['variantid', 'combined_text']], left_on='variantid2', right_on='variantid', how='left')
    test_data = test_data.rename(columns={'combined_text': 'text_2'})
    test_data = test_data.drop(columns=['variantid'])

    test_data = test_data.dropna()

    return test_data

def combine_embeddings(row):
    pic_embeddings = np.concatenate([row['pic_embeddings_1'][0], row['pic_embeddings_2'][0]])
    text_embeddings = np.concatenate([row['text_embedding_1'], row['text_embedding_2']])
    return np.concatenate([pic_embeddings, text_embeddings])

def prepare_test_data(test_data, tfidf_vectorizer):
    text_data = test_data['text_1'] + ' ' + test_data['text_2']
    text_embeddings = tfidf_vectorizer.transform(text_data).toarray()

    split_index = text_embeddings.shape[1] // 2
    test_data['text_embedding_1'] = list(text_embeddings[:, :split_index])
    test_data['text_embedding_2'] = list(text_embeddings[:, split_index:])

    test_data['combined_embeddings'] = test_data.apply(combine_embeddings, axis=1)

    X_test = np.vstack(test_data['combined_embeddings'].values)

    return X_test

def main():
    attributes, resnet, text_and_bert, test = load_test_data()
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
