import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Univariate Analysis

def plot_histogram(df, column, title="Histogram", bins=30):
    """
    Plot a histogram for a specific column.
    :param df: pandas DataFrame
    :param column: str, column name to plot
    :param title: str, title of the plot
    :param bins: int, number of bins
    """
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], bins=bins, kde=True)
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def plot_boxplot(df, column, title="Boxplot"):
    """
    Plot a boxplot for a specific column.
    :param df: pandas DataFrame
    :param column: str, column name to plot
    :param title: str, title of the plot
    """
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column])
    plt.title(title)
    plt.xlabel(column)
    plt.show()

# Bivariate Analysis

def plot_scatter(df, x_col, y_col, title="Scatter Plot"):
    """
    Plot a scatter plot between two columns.
    :param df: pandas DataFrame
    :param x_col: str, column name for the x-axis
    :param y_col: str, column name for the y-axis
    :param title: str, title of the plot
    """
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

def plot_correlation_heatmap(df, columns, title="Correlation Heatmap"):
    """
    Plot a heatmap to show correlations between multiple columns.
    :param df: pandas DataFrame
    :param columns: list, list of column names to include in the heatmap
    :param title: str, title of the plot
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[columns].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.show()

# Multivariate Analysis

def plot_pairplot(df, columns, title="Pairplot"):
    """
    Plot a pairplot for multiple columns to analyze relationships.
    :param df: pandas DataFrame
    :param columns: list, list of column names to include in the pairplot
    :param title: str, title of the plot
    """
    sns.pairplot(df[columns])
    plt.suptitle(title, y=1.02)
    plt.show()

# Example Usage of Updated Functions

def perform_visualizations(df):
    """
    Perform univariate, bivariate, and multivariate visualizations.
    :param df: pandas DataFrame
    """
    # Univariate Analysis
    plot_histogram(df, column="Total Data Volume", title="Total Data Volume Distribution")
    plot_boxplot(df, column="Dur. (ms)", title="Session Duration Distribution")

    # Bivariate Analysis
    plot_scatter(df, x_col="Total DL (Bytes)", y_col="Total UL (Bytes)", title="Download vs Upload")
    plot_correlation_heatmap(
        df,
        columns=[
            "Social Media DL (Bytes)", "Google DL (Bytes)", "Email DL (Bytes)",
            "Youtube DL (Bytes)", "Netflix DL (Bytes)", "Gaming DL (Bytes)", "Other DL (Bytes)"
        ],
        title="Application Data Correlation"
    )

    # Multivariate Analysis
    plot_pairplot(
        df,
        columns=[
            "Social Media DL (Bytes)", "Google DL (Bytes)", "Email DL (Bytes)",
            "Youtube DL (Bytes)", "Netflix DL (Bytes)", "Gaming DL (Bytes)", "Other DL (Bytes)"
        ],
        title="Multivariate Relationships"
    )


# Distributions per Handset Type
def distribution_by_handset(df, metric, title):
    """
    Plot and interpret the distribution of a metric per handset type.
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="handset_type", y=metric)
    plt.xticks(rotation=45)
    plt.title(title)
    plt.show()