class DataFrameMissingValueChecker:
    def __init__(self, df):
        """
        Initialize the DataFrameMissingValueChecker with a pandas DataFrame.
        
        Parameters:
        - df: The pandas DataFrame to analyze.
        """
        self.df = df

    def check_missing_values(self):
        """
        Check for missing values in the DataFrame and return a summary.

        Returns:
        - missing_summary: A pandas Series showing the count of missing values for each column.
        """
        missing_summary = self.df.isnull().sum()
        print("Missing values summary:")
        print(missing_summary)
        return missing_summary

    def calculate_missing_percent(self):
        """
        Calculate the percentage of missing values for each column in the DataFrame.

        Returns:
        - missing_percent: A pandas Series showing the percentage of missing values for each column.
        """
        missing_percent = (self.df.isnull().sum() / len(self.df)) * 100
        print("Percentage of missing values for each column:")
        print(missing_percent)
        return missing_percent

    def convert_bytes_to_megabytes(self, column_name):
        """
        Convert values from bytes to megabytes for a specified column in the DataFrame.

        Parameters:
        - column_name: The name of the column to convert.

        Returns:
        - DataFrame: The DataFrame with the specified column converted to megabytes.
        """
        if column_name in self.df.columns:
            self.df[column_name] = self.df[column_name] / (1024 * 1024)
            print(f"Converted {column_name} from bytes to megabytes.")
        else:
            print(f"Column {column_name} not found in DataFrame.")
        return self.df