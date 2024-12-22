import pandas as pd
from sklearn.decomposition import PCA


def aggregate_user_behavior(df):
    """
    Aggregate user behavior metrics.
    :param df: pandas DataFrame
    :return: aggregated DataFrame
    """
    user_behavior = df.groupby("imsi").agg(
        number_of_sessions=("bearer_id", "count"),
        total_duration=("dur._(ms)", "sum"),
        total_download=("total_dl_(bytes)", "sum"),
        total_upload=("total_ul_(bytes)", "sum"),
        total_data_volume=("total_data_volume", "sum"),
    ).reset_index()
    return user_behavior


def top_handsets(df, n=10):
    """
    Identify the top N handsets used by customers.
    :param df: pandas DataFrame
    :param n: int, number of top handsets to return
    :return: DataFrame
    """
    return df["handset_type"].value_counts().head(n)


def top_handset_manufacturers(df, n=3):
    """
    Identify the top N handset manufacturers.
    :param df: pandas DataFrame
    :param n: int, number of top manufacturers to return
    :return: DataFrame
    """
    return df["handset_manufacturer"].value_counts().head(n)


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
            df[df["handset_manufacturer"] == manufacturer]["handset_type"]
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
    df["decile"] = pd.qcut(df["total_duration"], 10, labels=False)
    decile_summary = df.groupby("decile").agg(
        total_data=("total_data_volume", "sum"),
        mean_duration=("total_duration", "mean"),
    ).reset_index()
    return df, decile_summary


def correlation_analysis(df):
    """
    Compute a correlation matrix for application-specific data.
    :param df: pandas DataFrame
    :return: correlation matrix
    """
    app_columns = [
        "social_media_dl_(bytes)", "google_dl_(bytes)", "email_dl_(bytes)",
        "youtube_dl_(bytes)", "netflix_dl_(bytes)", "gaming_dl_(bytes)", "other_dl_(bytes)"
    ]
    return df[app_columns].corr()


def dimensionality_reduction(df):
    """
    Perform PCA on application-specific data.
    :param df: pandas DataFrame
    :return: PCA results
    """
    app_columns = [
        "social_media_dl_(bytes)", "google_dl_(bytes)", "email_dl_(bytes)",
        "youtube_dl_(bytes)", "netflix_dl_(bytes)", "gaming_dl_(bytes)", "other_dl_(bytes)"
    ]
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(df[app_columns])
    explained_variance = pca.explained_variance_ratio_
    return principal_components, explained_variance


def perform_user_overview_analysis(df):
    """
    Perform a full user overview analysis.
    :param df: pandas DataFrame
    """
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
    principal_components, explained_variance = dimensionality_reduction(df)

    # Return analysis results
    return {
        "user_behavior": user_behavior,
        "top_handsets": top_handsets_df,
        "top_manufacturers": top_manufacturers_df,
        "top_handsets_per_mfg": top_handsets_per_mfg,
        "decile_summary": decile_summary,
        "correlation_matrix": correlation_matrix,
        "principal_components": principal_components,
        "explained_variance": explained_variance,
    }
