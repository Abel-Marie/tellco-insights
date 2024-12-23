import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


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



def train_regression_model(satisfaction_df, features, target="satisfaction_score"):
    """
    Train a regression model to predict the satisfaction score.
    :param satisfaction_df: pandas DataFrame, contains features and satisfaction scores
    :param features: list of str, column names to use as predictors
    :param target: str, column name of the target variable
    :return: trained model, test features, test targets, predictions
    """
    # Prepare data
    X = satisfaction_df[features]
    y = satisfaction_df[target]

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train regression model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Model Evaluation:\nMean Squared Error: {mse:.2f}\nR^2 Score: {r2:.2f}")

    return model, X_test, y_test, predictions


def cluster_users_by_satisfaction(satisfaction_df, k=2):
    """
    Perform k-means clustering on engagement and experience scores.
    :param satisfaction_df: pandas DataFrame containing engagement and experience scores
    :param k: int, number of clusters
    :return: DataFrame with cluster labels, k-means model
    """
    # Select features for clustering
    clustering_features = satisfaction_df[["engagement_score", "experience_score"]]

    # Fit K-Means model
    kmeans = KMeans(n_clusters=k, random_state=42)
    satisfaction_df["satisfaction_cluster"] = kmeans.fit_predict(clustering_features)

    return satisfaction_df, kmeans

def plot_clusters(satisfaction_df):
    """
    Visualize clusters using engagement and experience scores.
    :param satisfaction_df: pandas DataFrame containing cluster labels
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x="engagement_score",
        y="experience_score",
        hue="satisfaction_cluster",
        palette="viridis",
        data=satisfaction_df,
        s=100,
        alpha=0.7
    )
    plt.title("K-Means Clustering of Engagement and Experience Scores")
    plt.xlabel("Engagement Score")
    plt.ylabel("Experience Score")
    plt.legend(title="Cluster", loc="best")
    plt.show()


def aggregate_scores_by_cluster(satisfaction_df):
    """
    Compute average satisfaction score and average experience score for each cluster.
    :param satisfaction_df: pandas DataFrame containing cluster labels
    :return: DataFrame with aggregated scores by cluster
    """
    cluster_aggregates = satisfaction_df.groupby("satisfaction_cluster").agg(
        avg_satisfaction_score=("satisfaction_score", "mean"),
        avg_experience_score=("experience_score", "mean"),
        avg_engagement_score=("engagement_score", "mean"),
        cluster_size=("satisfaction_cluster", "size")
    ).reset_index()

    return cluster_aggregates
