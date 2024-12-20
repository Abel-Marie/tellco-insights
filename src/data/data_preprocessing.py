import pandas as pd
import numpy as np

def handle_missing_values(df, method="mean"):
    """
    Handle missing values in the dataset.
    :param df: pandas DataFrame
    :param method: str, method to handle missing values ('mean', 'median', 'mode', 'drop')
    :return: pandas DataFrame
    """
    if method == "mean":
        df = df.fillna(df.mean())
    elif method == "median":
        df = df.fillna(df.median())
    elif method == "mode":
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        df = df.fillna(df.mean())
    elif method == "drop":
        df = df.dropna()
    else:
        raise ValueError("Invalid method. Choose from 'mean', 'median', 'mode', or 'drop'.")
    return df

def detect_and_handle_outliers(df, cols, method="zscore", threshold=3):
    """
    Detect and handle outliers in specified columns.
    :param df: pandas DataFrame
    :param cols: list of column names to check for outliers
    :param method: str, method to detect outliers ('zscore', 'iqr')
    :param threshold: threshold for z-score or IQR
    :return: pandas DataFrame
    """
    for col in cols:
        if method == "zscore":
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < threshold]
        elif method == "iqr":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df = df[(df[col] >= Q1 - threshold * IQR) & (df[col] <= Q3 + threshold * IQR)]
        else:
            raise ValueError("Invalid method. Choose 'zscore' or 'iqr'.")
    return df

def add_total_data_volume(df):
    """
    Add a new column for total data volume (Download + Upload).
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    df["Total Data Volume"] = df["Total DL (Bytes)"] + df["Total UL (Bytes)"]
    return df
