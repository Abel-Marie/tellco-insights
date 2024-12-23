import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

def aggregate_experience_metrics(df):
    """
    Aggregate user experience metrics per customer (msisdn/number).
    :param df: pandas DataFrame
    :return: aggregated DataFrame
    """
    aggregated_df = df.groupby("msisdn/number").agg(
        avg_tcp_retransmission=("tcp_dl_retrans._vol_(bytes)", "mean"),
        avg_rtt=("avg_rtt_dl_(ms)", "mean"),
        avg_throughput=("avg_bearer_tp_dl_(kbps)", "mean"),
        handset_type=("handset_type", "first")
    ).reset_index()
    return aggregated_df

def top_bottom_frequent_values(df, column, n=10):
    """
    Compute the top, bottom, and most frequent values for a given column.
    :param df: pandas DataFrame
    :param column: str, column name
    :param n: int, number of values to return
    :return: dict with top, bottom, and frequent values
    """
    top_values = df.nlargest(n, column)[[column]]
    bottom_values = df.nsmallest(n, column)[[column]]
    frequent_values = df[column].value_counts().head(n)

    return {
        "top": top_values,
        "bottom": bottom_values,
        "frequent": frequent_values
    }

def plot_top_n_throughput_distribution_by_handset(df, top_n=20):
    """
    Plot the distribution of average throughput for the top N handset types by throughput.
    :param df: pandas DataFrame
    :param top_n: int, number of top handset types to include
    """
    # Get top N handset types by average throughput
    top_handsets = df.groupby("handset_type")["avg_throughput"].mean().nlargest(top_n).index
    filtered_df = df[df["handset_type"].isin(top_handsets)]

    plt.figure(figsize=(12, 6))
    sns.boxplot(x="handset_type", y="avg_throughput", data=filtered_df)
    plt.title(f"Throughput Distribution by Top {top_n} Handset Types")
    plt.xlabel("Handset Type")
    plt.ylabel("Average Throughput (kbps)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def plot_top_n_tcp_retransmission_by_handset(df, top_n=20):
    """
    Plot the average TCP retransmission for the top N handset types by retransmission.
    :param df: pandas DataFrame
    :param top_n: int, number of top handset types to include
    """
    # Get top N handset types by average TCP retransmission
    retransmission_summary = df.groupby("handset_type")["avg_tcp_retransmission"].mean().reset_index()
    top_handsets = retransmission_summary.nlargest(top_n, "avg_tcp_retransmission")
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x="handset_type", y="avg_tcp_retransmission", data=top_handsets)
    plt.title(f"Average TCP Retransmission by Top {top_n} Handset Types")
    plt.xlabel("Handset Type")
    plt.ylabel("Average TCP Retransmission (Bytes)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

def cluster_users_by_experience(df, k=3):
    """
    Perform k-means clustering on experience metrics.
    :param df: pandas DataFrame
    :param k: int, number of clusters
    :return: DataFrame with cluster labels and KMeans model
    """
    metrics = ["avg_tcp_retransmission", "avg_rtt", "avg_throughput"]
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(df[metrics].fillna(0))

    kmeans = KMeans(n_clusters=k, random_state=42)
    df["experience_cluster"] = kmeans.fit_predict(normalized_data)

    return df, kmeans

def describe_clusters(df):
    """
    Provide descriptions of experience clusters.
    :param df: pandas DataFrame
    :return: summary DataFrame
    """
    cluster_summary = df.groupby("experience_cluster").agg(
        avg_tcp_retransmission=("avg_tcp_retransmission", "mean"),
        avg_rtt=("avg_rtt", "mean"),
        avg_throughput=("avg_throughput", "mean"),
        count=("experience_cluster", "size")
    ).reset_index()

    return cluster_summary

def perform_experience_analysis(df):
    """
    Perform a complete experience analysis.
    :param df: pandas DataFrame
    :return: results dictionary
    """
    # Aggregate experience metrics
    aggregated_metrics = aggregate_experience_metrics(df)

    # Top, bottom, and frequent values for TCP retransmission, RTT, and throughput
    tcp_analysis = top_bottom_frequent_values(aggregated_metrics, "avg_tcp_retransmission")
    rtt_analysis = top_bottom_frequent_values(aggregated_metrics, "avg_rtt")
    throughput_analysis = top_bottom_frequent_values(aggregated_metrics, "avg_throughput")

    # Visualizations
    plot_top_n_throughput_distribution_by_handset(aggregated_metrics)
    plot_top_n_tcp_retransmission_by_handset(aggregated_metrics)

    # Clustering users by experience
    clustered_data, kmeans_model = cluster_users_by_experience(aggregated_metrics)

    # Describe clusters
    cluster_summary = describe_clusters(clustered_data)

    return {
        "aggregated_metrics": aggregated_metrics,
        "tcp_analysis": tcp_analysis,
        "rtt_analysis": rtt_analysis,
        "throughput_analysis": throughput_analysis,
        "clustered_data": clustered_data,
        "cluster_summary": cluster_summary,
        "kmeans_model": kmeans_model
    }
