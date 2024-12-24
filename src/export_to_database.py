from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

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