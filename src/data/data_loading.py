 import pandas as pd
import os
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Fetch database connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 5432))  # Default port for PostgreSQL
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




def load_data_from_postgres(query):
    """
    Load data from PostgreSQL using psycopg2.
    :param query: SQL query string
    :return: pandas DataFrame
    """
    try:
        if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD]):
            raise ValueError("One or more database credentials are missing!")

        # Establish a connection to the database
        connection = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        # Load the data using pandas
        df = pd.read_sql_query(query, connection)
        connection.close()  # Close the connection
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None



def load_data_using_sql_alchemy(query):
    """
    Load data from PostgreSQL using SQLAlchemy.
    :param query: SQL query string
    :return: pandas DataFrame
    """
    try: 
        # Create a connection string
        connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        
        # Create an SQLAlchemy engine
        engine = create_engine(connection_string)
        
        # Load data into pandas DataFrame
        df = pd.read_sql_query(query, engine)
        engine.dispose()  # Dispose of the engine
        return df 
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
