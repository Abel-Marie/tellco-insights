import pandas as pd
from src.data.data_preprocessing import handle_missing_values, detect_and_handle_outliers, add_total_data_volume
from src.analysis.data_visualization import plot_histogram, plot_scatter, plot_correlation_heatmap

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
    top_manufacturers = df["Last Location Name"].value_counts().head(n)
    return top_manufacturers

from src.analysis.data_visualization import plot_bar_chart, plot_line_chart, plot_boxplot

def calculate_average_session_duration(df):
    """
    Calculate the average session duration per user.
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    avg_duration = df.groupby("IMSI").agg(
        average_session_duration=("Dur. (ms)", "mean")
    ).reset_index()
    print("\nAverage Session Duration Per User:")
    print(avg_duration.head())
    return avg_duration

def calculate_data_usage_trends(df):
    """
    Calculate daily data usage trends.
    :param df: pandas DataFrame
    :return: pandas DataFrame
    """
    df["Start Date"] = pd.to_datetime(df["Start"]).dt.date
    data_trends = df.groupby("Start Date").agg(
        total_download=("Total DL (Bytes)", "sum"),
        total_upload=("Total UL (Bytes)", "sum"),
        total_data_volume=("Total Data Volume", "sum")
    ).reset_index()
    print("\nData Usage Trends:")
    print(data_trends.head())
    return data_trends


def perform_user_overview_analysis(df):
    """
    Perform a full user overview analysis with enhanced visualizations and insights.
    :param df: pandas DataFrame
    """
    # Handle missing values
    df = handle_missing_values(df)
    
    # Add total data volume column
    df = add_total_data_volume(df)

    # Aggregate user behavior
    user_behavior = aggregate_user_behavior(df)

    # Top handsets and manufacturers
    top_handsets_df = top_handsets(df)
    top_manufacturers_df = top_handset_manufacturers(df)

    # Additional Insights
    avg_session_duration = calculate_average_session_duration(df)
    data_usage_trends = calculate_data_usage_trends(df)

    # Visualizations
    # Histogram for total data volume
    plot_histogram(user_behavior, "total_data_volume", title="Total Data Volume Distribution")

    # Bar chart for top handsets
    plot_bar_chart(
        data=top_handsets_df.reset_index().rename(columns={"index": "Handset", "IMEI": "Count"}),
        x_col="Handset", y_col="Count",
        title="Top 10 Handsets"
    )

    # Line chart for data usage trends
    plot_line_chart(data=data_usage_trends, x_col="Start Date", y_col="total_data_volume", title="Daily Data Usage Trends")

    # Boxplot for average session duration
    plot_boxplot(data=avg_session_duration, x_col="IMSI", y_col="average_session_duration", title="Average Session Duration Per User")

    return user_behavior, top_handsets_df, top_manufacturers_df
