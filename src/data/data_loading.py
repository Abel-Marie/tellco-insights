import pandas as pd
import os
import sqlite3

# Define the path to the dataset folder
DATASET_FOLDER = os.path.join(os.getcwd(), "dataset")

def load_csv(file_name):
    """
    Load a CSV file from the dataset folder.
    :param file_name: Name of the CSV file (str)
    :return: pandas DataFrame
    """
    file_path = os.path.join(DATASET_FOLDER, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_name}' not found in {DATASET_FOLDER}")
    return pd.read_csv(file_path)

def load_excel(file_name, sheet_name=None):
    """
    Load an Excel file from the dataset folder.
    :param file_name: Name of the Excel file (str)
    :param sheet_name: Sheet name to load (str or None for all sheets)
    :return: pandas DataFrame or dict of DataFrames if sheet_name is None
    """
    file_path = os.path.join(DATASET_FOLDER, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_name}' not found in {DATASET_FOLDER}")
    return pd.read_excel(file_path, sheet_name=sheet_name)

def load_sql_to_dataframe(db_file, sql_file):
    """
    Load SQL data into a DataFrame using SQLite.
    :param db_file: Name of the SQLite database file (str)
    :param sql_file: Name of the SQL script file (str)
    :return: SQLite connection object
    """
    db_path = os.path.join(DATASET_FOLDER, db_file)
    sql_path = os.path.join(DATASET_FOLDER, sql_file)
    
    if not os.path.exists(sql_path):
        raise FileNotFoundError(f"SQL script '{sql_file}' not found in {DATASET_FOLDER}")
    
    # Connect to SQLite database
    conn = sqlite3.connect(db_path)
    with open(sql_path, "r") as f:
        sql_script = f.read()
    conn.executescript(sql_script)
    
    return conn  # Return the connection object for further querying

def query_sql(conn, query):
    """
    Query data from an SQLite database.
    :param conn: SQLite connection object
    :param query: SQL query string
    :return: pandas DataFrame
    """
    return pd.read_sql_query(query, conn)
