import os
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
import pandas as pd

# Load environment variables from the .env file
load_dotenv()

# Fetch database connection parameters from environment variables
DB_HOST = os.getenv("DB_HOST")
DB_PORT = int(os.getenv("DB_PORT", 5432))  
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")



def export_to_postgres(df, table_name):
    """
    Export a DataFrame to a PostgreSQL database.
    :param df: pandas DataFrame to export
    :param table_name: str, name of the database table
    """
    try:
        # Create a connection to the PostgreSQL database
        engine = create_engine(f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
        
        # Export the DataFrame to the PostgreSQL table
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        print(f"Table '{table_name}' successfully exported to PostgreSQL database.")
    
    except Exception as e:
        print(f"Error exporting table: {e}")
        

# Main function to prepare and export the final table
def prepare_and_export_final_table(df, table_name):
    """
    Prepare and export the final table to PostgreSQL.
    :param df: pandas DataFrame containing user IDs, engagement score, experience score, and satisfaction score
    :param table_name: str, name of the database table
    """
    # Select relevant columns
    final_table = df[["msisdn/number", "engagement_score", "experience_score", "satisfaction_score"]]
    
    # Rename columns for clarity
    final_table = final_table.rename(columns={
        "msisdn/number": "user_id",
        "engagement_score": "engagement_score",
        "experience_score": "experience_score",
        "satisfaction_score": "satisfaction_score"
    })
    
    # Export to PostgreSQL
    export_to_postgres(final_table, table_name)
