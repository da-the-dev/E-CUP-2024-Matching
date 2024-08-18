import pandas as pd
import numpy as np
import argparse
import json
from sklearn.metrics import precision_recall_curve, auc


class ParticipantVisibleError(Exception):
    pass


def validate(target_df: pd.DataFrame, prediction_df: pd.DataFrame) -> bool:
    target_df = target_df.drop_duplicates(["variantid1", "variantid2"])

    shape_ok = target_df.shape[0] == prediction_df.shape[0]
    nan_ok = prediction_df["target_pred"].notna().all()

    not_bool_target = len(prediction_df["target_pred"].unique()) == 2

    if not shape_ok:
        print("Shape mismatch: target_df has", target_df.shape[0], "rows, prediction_df has", prediction_df.shape[0], "rows")
    if not nan_ok:
        print("NaN values found in target_pred column")
    if not not_bool_target:
        print("target_pred column has less than 3 unique values:", prediction_df["target_pred"].unique())

    return shape_ok and nan_ok and not_bool_target


def calculate_macro_prauc_by_category(df: pd.DataFrame, categories: np.ndarray) -> float:
    pr_auc_by_category = []

    for category in categories:
        cat_indices = df["category2"] == category
        y_true = df.loc[cat_indices, "target_true"]
        y_scores = df.loc[cat_indices, "target_pred"]

        if len(y_true) == 0 or sum(y_true) == 0:
            pr_auc_by_category.append(0)
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_scores)

        pr_auc = auc(recall, precision)
        pr_auc_by_category.append(pr_auc)

    if len(pr_auc_by_category) == 0:
        return 0.0

    macro_prauc = np.mean(pr_auc_by_category)
    return macro_prauc


def compare_csv_files_with_categories(csv_file1: str, csv_file2: str):
    submission = pd.read_csv(csv_file2)
    test_labels = pd.read_csv(csv_file1)

    df = test_labels.merge(submission, on=["variantid1", "variantid2"], suffixes=('_true', '_pred'))

    if not validate(test_labels, df):
        raise ParticipantVisibleError("Validation failed for input dataframes")

    categories = test_labels['category2'].dropna().unique()

    macro_prauc = calculate_macro_prauc_by_category(df, categories)

    return macro_prauc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--public_test_url', type=str, required=True)
    parser.add_argument('--public_prediction_url', type=str, required=True)
    args = parser.parse_args()

    macro_prauc = compare_csv_files_with_categories(args.public_test_url, args.public_prediction_url)
    
    print(f'score: {macro_prauc}')
