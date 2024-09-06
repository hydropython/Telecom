import psycopg2
import pandas as pd

class PostgresDataLoader:
    def __init__(self, host, database, user, password):
        """
        Initialize the PostgresDataLoader with connection details.
        
        Parameters:
        - host: PostgreSQL server address
        - database: Name of the database to connect to
        - user: PostgreSQL user
        - password: PostgreSQL user password
        """
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.connection = None

    def connect(self):
        """Establish a connection to the PostgreSQL database."""
        try:
            self.connection = psycopg2.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self.password
            )
            print("Connection to PostgreSQL DB successful")
        except psycopg2.OperationalError as e:
            print(f"Connection error: {e}")
            self.connection = None

    def load_data(self, query):
        """
        Load data from PostgreSQL into a pandas DataFrame by executing a SQL query.

        Parameters:
        - query: SQL query to execute

        Returns:
        - DataFrame: pandas DataFrame containing the query result
        """
        if self.connection is None:
            print("No active database connection. Call connect() first.")
            return None
        
        try:
            df = pd.read_sql_query(query, self.connection)
            print("Data successfully loaded into DataFrame")
            return df
        except Exception as e:
            print(f"Error executing query: {e}")
            return None

    def close(self):
        """Close the PostgreSQL connection."""
        if self.connection:
            self.connection.close()
            print("PostgreSQL connection is closed")