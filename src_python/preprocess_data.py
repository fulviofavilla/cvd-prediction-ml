import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy import stats
import os
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def run_preprocessing(input_path: str, train_out: str, test_out: str):
    """
    Executes the data cleaning and splitting process according to the TCC methodology.
    """
    if not os.path.exists(input_path):
        logging.error(f"Raw data file not found: {input_path}")
        return

    logging.info("Loading raw dataset...")
    # The original dataset uses ';' as a delimiter
    df = pd.read_csv(input_path, delimiter=';')

    # 1. Drop irrelevant identifier
    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    # 2. Handle missing values (Mean imputation)
    df.fillna(df.mean(), inplace=True)

    # 3. Class Balancing (Upsampling the minority class)
    logging.info("Balancing classes via upsampling...")
    df_majority = df[df['cardio'] == 0]
    df_minority = df[df['cardio'] == 1]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )
    df_balanced = pd.concat([df_majority, df_minority_upsampled])

    X = df_balanced.drop('cardio', axis=1)
    y = df_balanced['cardio']

    # 4. Train/Test Split (80/20)
    logging.info("Splitting dataset into 80% Train and 20% Test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # 5. Outlier Removal (Train Set ONLY)
    logging.info("Filtering outliers using Z-score threshold (< 3)...")
    numeric_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
    z_scores = stats.zscore(X_train[numeric_cols])
    abs_z_scores = np.abs(z_scores)

    # Keep only rows where all numeric features are within 3 standard deviations
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    X_train_final = X_train[filtered_entries]
    y_train_final = y_train[filtered_entries]

    total_records = len(X_train_final) + len(X_test)
    logging.info(f"Preprocessing complete. Total valid records for modeling: {total_records}")

    # 6. Export to final CSVs
    train_df = pd.concat([X_train_final, y_train_final], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(train_out), exist_ok=True)

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)
    logging.info(f"Preprocessed datasets saved to data/ directory.")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    raw_csv = os.path.join(base_dir, "../data/cardio_raw.csv")
    train_csv = os.path.join(base_dir, "../data/train_final.csv")
    test_csv = os.path.join(base_dir, "../data/test_final.csv")

    run_preprocessing(raw_csv, train_csv, test_csv)
