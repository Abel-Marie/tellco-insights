import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_user_behavior(df):
    """
    Aggregate user behavior metrics.
    :param df: pandas DataFrame
    :return: aggregated DataFrame
    """
    user_behavior = df.groupby("IMSI").agg(
        number_of_sessions=("Bearer Id", "count"),
        total_duration=("Dur. (ms)", "sum"),
        total_download=("Total DL (Bytes)", "sum"),
        total_upload=("Total UL (Bytes)", "sum"),
        total_data_volume=("Total Data Volume", "sum")
    ).reset_index()
    return user_behavior

def top_handsets(df, n=10):
    """
    Identify the top N handsets used by customers.
    :param df: pandas DataFrame
    :param n: int, number of top handsets to return
    :return: DataFrame
    """
    top_handsets = df["IMEI"].value_counts().head(n)
    return top_handsets

def top_handset_manufacturers(df, n=3):
    """
    Identify the top N handset manufacturers.
    :param df: pandas DataFrame
    :param n: int, number of top manufacturers to return
    :return: DataFrame
    """
    top_manufacturers = df["Handset Manufacturer"].value_counts().head(n)
    return top_manufacturers

def top_handsets_per_manufacturer(df, manufacturers, n=5):
    """
    Identify the top N handsets per manufacturer.
    :param df: pandas DataFrame
    :param manufacturers: list, top manufacturers
    :param n: int, number of top handsets to return per manufacturer
    :return: dict of DataFrames
    """
    results = {}
    for manufacturer in manufacturers:
        top_handsets = (
            df[df["Handset Manufacturer"] == manufacturer]["IMEI"]
            .value_counts()
            .head(n)
        )
        results[manufacturer] = top_handsets
    return results

def segment_users(df):
    """
    Segment users into deciles based on total session duration.
    :param df: pandas DataFrame
    :return: DataFrame with decile segmentation
    """
    df["Decile"] = pd.qcut(df["total_duration"], 10, labels=False)
    decile_summary = df.groupby("Decile").agg(
        total_data=("total_data_volume", "sum"),
        mean_duration=("total_duration", "mean")
    ).reset_index()
    return df, decile_summary

def correlation_analysis(df):
    """
    Compute a correlation matrix for application-specific data.
    :param df: pandas DataFrame
    :return: correlation matrix
    """
    app_columns = [
        "Social Media DL (Bytes)", "Google DL (Bytes)", "Email DL (Bytes)",
        "Youtube DL (Bytes)", "Netflix DL (Bytes)", "Gaming DL (Bytes)", "Other DL (Bytes)"
    ]
    correlation_matrix = df[app_columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Application Data Correlation Matrix")
    plt.show()
    return correlation_matrix

def dimensionality_reduction(df):
    """
    Perform Principal Component Analysis (PCA) on application-specific data.
    :param df: pandas DataFrame
    :return: PCA results
    """
    app_columns = [
        "Social Media DL (Bytes)", "Google DL (Bytes)", "Email DL (Bytes)",
        "Youtube DL (Bytes)", "Netflix DL (Bytes)", "Gaming DL (Bytes)", "Other DL (Bytes)"
    ]
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df[app_columns].fillna(0))
    explained_variance = pca.explained_variance_ratio_

    plt.figure(figsize=(8, 6))
    plt.scatter(principal_components[:, 0], principal_components[:, 1], alpha=0.7)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA - Application Data")
    plt.show()

    return explained_variance

def perform_user_overview_analysis(df):
    """
    Perform a full user overview analysis with enhanced visualizations and insights.
    :param df: pandas DataFrame
    """
    from src.data.data_preprocessing import add_total_data_volume

    # Add total data volume column
    df = add_total_data_volume(df)

    # Aggregate user behavior
    user_behavior = aggregate_user_behavior(df)

    # Top handsets and manufacturers
    top_handsets_df = top_handsets(df)
    top_manufacturers_df = top_handset_manufacturers(df)
    top_manufacturer_list = top_manufacturers_df.index.tolist()

    # Top handsets per top manufacturers
    top_handsets_per_mfg = top_handsets_per_manufacturer(df, top_manufacturer_list)

    # Segment users into deciles
    user_behavior, decile_summary = segment_users(user_behavior)

    # Correlation Analysis
    correlation_matrix = correlation_analysis(df)

    # Dimensionality Reduction (PCA)
    explained_variance = dimensionality_reduction(df)

    # Visualizations
    print("Top 10 Handsets:", top_handsets_df)
    print("Top 3 Manufacturers:", top_manufacturers_df)
    print("Top Handsets per Manufacturer:", top_handsets_per_mfg)
    print("Decile Summary:", decile_summary)
    print("Explained Variance from PCA:", explained_variance)

    return {
        "user_behavior": user_behavior,
        "top_handsets": top_handsets_df,
        "top_manufacturers": top_manufacturers_df,
        "top_handsets_per_mfg": top_handsets_per_mfg,
        "decile_summary": decile_summary,
        "correlation_matrix": correlation_matrix,
        "explained_variance": explained_variance,
    }
