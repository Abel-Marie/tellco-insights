import matplotlib.pyplot as plt
import seaborn as sns

def plot_histogram(df, column, bins=50, title=None):
    """
    Plot a histogram for a specific column.
    :param df: pandas DataFrame
    :param column: column name
    :param bins: number of bins
    :param title: title for the plot
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[column], bins=bins, kde=True)
    plt.title(title or f"Histogram of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.show()

def plot_scatter(df, x_col, y_col, alpha=0.5, title=None):
    """
    Plot a scatterplot for two columns.
    :param df: pandas DataFrame
    :param x_col: x-axis column
    :param y_col: y-axis column
    :param alpha: transparency of points
    :param title: title for the plot
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, alpha=alpha)
    plt.title(title or f"{x_col} vs {y_col}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()

def plot_correlation_heatmap(df, columns=None, title="Correlation Heatmap"):
    """
    Plot a heatmap for the correlation matrix of selected columns.
    :param df: pandas DataFrame
    :param columns: list of column names to include in the correlation matrix
    :param title: title for the plot
    """
    if columns:
        df = df[columns]
    correlation_matrix = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title(title)
    plt.show()
