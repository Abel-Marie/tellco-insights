import pandas as pd
import numpy as np

def handle_missing_values(df, method="mean"):
    """
    Handle missing values in the dataset.
    :param df: pandas DataFrame
    :param method: str, method to handle missing values ('mean', 'median', 'mode', 'drop')
    :return: pandas DataFrame
    """
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Apply method only to numeric columns
            if method == "mean":
                df[column] = df[column].fillna(df[column].mean())
            elif method == "median":
                df[column] = df[column].fillna(df[column].median())
        elif method == "mode":
            # Fill with mode for both numeric and non-numeric columns
            df[column] = df[column].fillna(df[column].mode().iloc[0])
        elif method == "drop":
            # Drop rows with missing values
            df = df.dropna()
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
