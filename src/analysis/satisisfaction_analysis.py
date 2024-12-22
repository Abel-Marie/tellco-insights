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
