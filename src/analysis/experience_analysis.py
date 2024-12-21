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
