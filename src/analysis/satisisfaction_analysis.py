import numpy as np
from scipy.spatial.distance import euclidean

def assign_engagement_score(df, engagement_clusters, least_engaged_cluster):
    """
    Assign engagement scores to each user based on Euclidean distance 
    from the least engaged cluster centroid.
    :param df: pandas DataFrame, engagement data
    :param engagement_clusters: k-means model for engagement clustering
    :param least_engaged_cluster: int, cluster representing least engaged users
    :return: DataFrame with Engagement Scores
    """
    centroids = engagement_clusters.cluster_centers_
    least_engaged_centroid = centroids[least_engaged_cluster]

    df["engagement_score"] = df[["sessions_frequency", "total_duration", "total_traffic"]].apply(
        lambda row: euclidean(row, least_engaged_centroid), axis=1
    )
    return df


def assign_experience_score(df, experience_clusters, worst_experience_cluster):
    """
    Assign experience scores to each user based on Euclidean distance 
    from the worst experience cluster centroid.
    :param df: pandas DataFrame, experience data
    :param experience_clusters: k-means model for experience clustering
    :param worst_experience_cluster: int, cluster representing the worst experience
    :return: DataFrame with Experience Scores
    """
    centroids = experience_clusters.cluster_centers_
    worst_experience_centroid = centroids[worst_experience_cluster]

    df["experience_score"] = df[["avg_tcp_retransmission", "avg_rtt", "avg_throughput"]].apply(
        lambda row: euclidean(row, worst_experience_centroid), axis=1
    )
    return df

def calculate_satisfaction_score(engagement_df, experience_df):
    """
    Calculate the satisfaction score as the average of engagement and experience scores.
    :param engagement_df: pandas DataFrame, contains engagement scores
    :param experience_df: pandas DataFrame, contains experience scores
    :return: DataFrame with satisfaction scores
    """
    # Merge engagement and experience data on the user identifier
    merged_df = engagement_df.merge(
        experience_df[["msisdn/number", "experience_score"]],
        on="msisdn/number",
        how="inner"
    )

    # Calculate the satisfaction score
    merged_df["satisfaction_score"] = (merged_df["engagement_score"] + merged_df["experience_score"]) / 2

    return merged_df

def top_satisfied_customers(satisfaction_df, n=10):
    """
    Retrieve the top N most satisfied customers based on satisfaction score.
    :param satisfaction_df: pandas DataFrame, contains satisfaction scores
    :param n: int, number of top customers to return
    :return: DataFrame of top N satisfied customers
    """
    return satisfaction_df.nlargest(n, "satisfaction_score")
