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

def get_top_n(df, column, n=10, ascending=False):
    """
    Get the top N rows based on a specific column.
    :param df: pandas DataFrame
    :param column: column name to sort by
    :param n: number of rows to return
    :param ascending: sort order
    :return: pandas DataFrame with top N rows
    """
    top_n = df.nlargest(n, column) if not ascending else df.nsmallest(n, column)
    print(f"\nTop {n} rows by {column}:")
    print(top_n)
    return top_n
