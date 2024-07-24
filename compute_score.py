import pandas as pd
from sklearn.metrics import roc_auc_score

def compare_csv_files(csv_file1, csv_file2, max_fpr=0.25):
    df1 = pd.read_csv(csv_file1)
    df2 = pd.read_csv(csv_file2)

    assert set(zip(df1['variantid1'], df1['variantid2'])) == set(zip(df2['variantid1'], df2['variantid2']))

    df1 = df1.sort_values(by=['variantid1', 'variantid2']).reset_index(drop=True)
    df2 = df2.sort_values(by=['variantid1', 'variantid2']).reset_index(drop=True)

    labels1 = df1['target'].tolist()
    labels2 = df2['target'].tolist()

    pauc = roc_auc_score(labels1, labels2, max_fpr=max_fpr)

    return pauc

if __name__ == "__main__":
    CSV_FILE_1 = 'data/submission.csv'
    CSV_FILE_2 = 'data/private_info/test.csv'

    pauc = compare_csv_files(CSV_FILE_1, CSV_FILE_2)
    print(f'PAUC: {pauc:.4f}')
