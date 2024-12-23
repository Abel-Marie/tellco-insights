import pandas as pd

def summarize_dataset(df):
    """
    Print basic statistics and information about the dataset.
    :param df: pandas DataFrame
    """
    print("\nDataset Info:")
    print(df.info())
    print("\nBasic Statistics:")
    print(df.describe())

def identify_missing_values(df):
    """
    Identify missing values in the dataset.
    :param df: pandas DataFrame
    :return: pandas DataFrame showing count of missing values per column
    """
    missing_values = df.isnull().sum()
    print("\nMissing Values:")
    print(missing_values[missing_values > 0])
    return missing_values

