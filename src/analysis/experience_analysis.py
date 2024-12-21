import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib.pyplot as plt
from seaborn as sns



# Aggregate User Experiance Metrics

def aggregate_experience_metrics(df):
    """
    Agregate Per Customer the average TCP 
    """
    aggrerated = df.groupby("MSIDN/Number").agg(
        avg_tcp_retransmission = ("TCP DL Retrans. Vol (Bytes)", "mean"),
        avg_rtt = ("Avg RTT DL (ms)", "mean"),
        handset_type = ("Handset Type", lambda x: x.mode()[0]),
        avg_throughput = ("Avg Bearer TP DL (kbps)", "mean", )
    ).reset_ndex()
    return aggrerated

# Top, Bottom, and Most Frequent Metrics
def compute_top_bottom_frequent(df, column, n=10):
    """
    Compute and return the top, bottom, and most frequent values of a column
    """
    top_values = df[column].nlargest(n)
    bottom_values = df[column].nsmallest(n)
    frequent_values = df[column].value_counts().head(n)
    return top_values, bottom_values, frequent_values

# K-Means Clustering

def Kmeans_clustering(df, features, k=3):
    """
    Perform K-Means clustering on selected features
    """
    scaler =MinMaxScaler()
    scaled_features = scaler.fit_transform(df[features])
    Kmeans = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = Kmeans.fit_predict(scaled_features)
    
    return df, Kmeans
    