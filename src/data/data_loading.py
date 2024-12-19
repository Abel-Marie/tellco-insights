import pandas as pd
import os
import sqlite3
from sqlalchemy import create_engine

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

# def load_excel(file_name, sheet_name=None):
#     """
#     Load an Excel file from the dataset folder.
#     :param file_name: Name of the Excel file (str)
#     :param sheet_name: Sheet name to load (str or None for all sheets)
#     :return: pandas DataFrame or dict of DataFrames if sheet_name is None
#     """
#     file_path = os.path.join(DATASET_FOLDER, file_name)
#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"File '{file_name}' not found in {DATASET_FOLDER}")
#     return pd.read_excel(file_path, sheet_name=sheet_name)


# Define the path to the dataset folder
DATASET_FOLDER = os.path.join(os.getcwd(), "dataset")

def connect_to_postgres(db_name, user, password, host="localhost", port=5432):
    """
    Connect to a PostgreSQL database.
    :param db_name: Name of the PostgreSQL database
    :param user: PostgreSQL username
    :param password: PostgreSQL password
    :param host: Host of the PostgreSQL server (default: localhost)
    :param port: Port of the PostgreSQL server (default: 5432)
    :return: SQLAlchemy engine
    """
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{db_name}")
    return engine

def query_postgres(engine, query):
    """
    Query data from a PostgreSQL database.
    :param engine: SQLAlchemy engine object
    :param query: SQL query string
    :return: pandas DataFrame
    """
    with engine.connect() as connection:
        return pd.read_sql_query(query, connection)
