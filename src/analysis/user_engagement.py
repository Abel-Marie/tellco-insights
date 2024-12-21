import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Task 2.1: Aggregate Metrics

def aggregate_user_engagement(df):
    """
    Aggregate engagement metrics per customer (MSISDN/Number).
    :param df: pandas DataFrame
    :return: aggregated DataFrame
    """
    aggregated_metrics = df.groupby("MSISDN/Number").agg(
        sessions_frequency=("Bearer Id", "count"),
        total_duration=("Dur. (ms)", "sum"),
        total_traffic=("Total Data Volume", "sum")
    ).reset_index()
    return aggregated_metrics

# Top 10 Customers by Engagement Metric

def top_customers_by_metric(aggregated_df, metric, n=10):
    """
    Retrieve top N customers by a specific engagement metric.
    :param aggregated_df: pandas DataFrame
    :param metric: str, metric column name
    :param n: int, number of top customers to return
    :return: pandas DataFrame
    """
    return aggregated_df.nlargest(n, metric)

# Normalize Metrics and Run K-Means Clustering

def normalize_and_cluster(aggregated_df, metrics, k=3):
    """
    Normalize metrics and run k-means clustering.
    :param aggregated_df: pandas DataFrame
    :param metrics: list, list of metric column names to normalize
    :param k: int, number of clusters
    :return: DataFrame with cluster labels, k-means model, and scaler
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(aggregated_df[metrics])

    kmeans = KMeans(n_clusters=k, random_state=42)
    aggregated_df["Cluster"] = kmeans.fit_predict(normalized_data)

    return aggregated_df, kmeans, scaler

# Metrics Summary per Cluster

def cluster_metrics_summary(aggregated_df, metrics):
    """
    Compute minimum, maximum, average, and total metrics for each cluster.
    :param aggregated_df: pandas DataFrame
    :param metrics: list, list of metric column names
    :return: pandas DataFrame summarizing metrics per cluster
    """
    summary = aggregated_df.groupby("Cluster")[metrics].agg([
        ("min", "min"),
        ("max", "max"),
        ("mean", "mean"),
        ("sum", "sum")
    ])
    return summary

# Aggregate Traffic per Application and Identify Top Users

def top_users_per_application(df, app_columns, n=10):
    """
    Aggregate total traffic per application and derive the top N users per application.
    :param df: pandas DataFrame
    :param app_columns: list, list of application-specific columns
    :param n: int, number of top users to return
    :return: dict of DataFrames for each application
    """
    top_users = {}
    for app in app_columns:
        aggregated_app = df.groupby("MSISDN/Number").agg({app: "sum"}).reset_index()
        top_users[app] = aggregated_app.nlargest(n, app)
    return top_users

# Plot Top 3 Most Used Applications

def plot_top_applications(df, app_columns):
    """
    Plot the top 3 most used applications.
    :param df: pandas DataFrame
    :param app_columns: list, list of application-specific columns
    """
    total_traffic = {app: df[app].sum() for app in app_columns}
    sorted_apps = sorted(total_traffic.items(), key=lambda x: x[1], reverse=True)[:3]

    apps, traffic = zip(*sorted_apps)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(apps), y=list(traffic))
    plt.title("Top 3 Most Used Applications")
    plt.xlabel("Applications")
    plt.ylabel("Total Traffic (Bytes)")
    plt.show()

# Elbow Method for Optimized k

def elbow_method(aggregated_df, metrics, max_k=10):
    """
    Determine the optimized value of k using the elbow method.
    :param aggregated_df: pandas DataFrame
    :param metrics: list, list of metric column names to normalize
    :param max_k: int, maximum number of clusters to test
    :return: None (plots the elbow curve)
    """
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(aggregated_df[metrics])

    distortions = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_data)
        distortions.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), distortions, marker="o")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Distortion")
    plt.show()
