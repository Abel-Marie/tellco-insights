import pandas as pd
import os
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Fetch database connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", 5432)  # Default port for PostgreSQL
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

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


def connect_to_postgres():
    """
    Establish a connection to a PostgreSQL database.
    :return: SQLAlchemy engine
    """
    try:
        engine = create_engine(
            f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        )
        # Test the connection
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        print("Successfully connected to the PostgreSQL database.")
        return engine
    except SQLAlchemyError as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        raise


def query_postgres(engine, query):
    """
    Query data from a PostgreSQL database.
    :param engine: SQLAlchemy engine object
    :param query: SQL query string
    :return: pandas DataFrame
    """
    try:
        with engine.connect() as connection:
            return pd.read_sql_query(query, connection)
    except SQLAlchemyError as e:
        print(f"Error executing query: {e}")
        raise


def load_data_from_database(table_name):
    """
    Load data from a specific table in the PostgreSQL database.
    :param table_name: Name of the table to load data from
    :return: pandas DataFrame
    """
    query = f"SELECT * FROM {table_name}"
    try:
        engine = connect_to_postgres()
        df = query_postgres(engine, query)
        print(f"Successfully loaded data from table '{table_name}'.")
        return df
    except SQLAlchemyError as e:
        print(f"Error loading data from table '{table_name}': {e}")
        raise
    finally:
        engine.dispose()  # Close the connection



