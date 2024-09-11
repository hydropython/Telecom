
import pandas as pd
import numpy as np  # Make sure to import NumPy


class DataFrameMissingValueChecker:
    """
    Class for checking missing values and performing basic data cleaning operations.
    """

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
        return missing_summary

    def calculate_missing_percent(self):
        """
        Calculate the percentage of missing values for each column in the DataFrame.

        Returns:
        - missing_percent: A pandas Series showing the percentage of missing values for each column.
        """
        missing_percent = (self.df.isnull().sum() / len(self.df)) * 100
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

    def fill_missing_numerical(self, method='mean', value=None):
        """
        Fill missing numerical data in all numerical columns using the chosen method.
        
        Args:
        method (str): The method to use for filling missing values. Options are 'mean', 'median', or 'value'.
        value (float/int): If method is 'value', this specifies the value to fill.
        """
        # Get all numerical columns
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column_name in numerical_columns:
            if method == 'mean':
                fill_value = self.df[column_name].mean()
            elif method == 'median':
                fill_value = self.df[column_name].median()
            elif method == 'value' and value is not None:
                fill_value = value
            else:
                raise ValueError("Invalid method or missing value for 'value' method.")
            
            self.df[column_name].fillna(fill_value, inplace=True)
            print(f"Filled missing values in column '{column_name}' using method '{method}' with value {fill_value}.")
            
        return self.df

    def fill_missing_and_undefined(self, columns):
        """
        Replace 'undefined' and NaN values with 'Unknown' in specified columns.
        
        Parameters:
        - columns: A list of column names where missing and 'undefined' values should be replaced.
        """
        for column_name in columns:
            if column_name in self.df.columns:
                self.df[column_name].replace('undefined', pd.NA, inplace=True)
                self.df[column_name].fillna('Unknown', inplace=True)
                print(f"Replaced 'undefined' and NaN values in '{column_name}' with 'Unknown'.")
            else:
                print(f"Column '{column_name}' not found in DataFrame.")
        
        return self.df

    def drop_columns(self, columns):
        """
        Drop specified columns from the DataFrame and return a new DataFrame with the dropped columns saved separately.

        Parameters:
        - columns: A list of column names to drop.

        Returns:
        - df_dropped: A new DataFrame with the specified columns removed.
        - dropped_columns: A DataFrame containing only the dropped columns with NaN values.
        """
        # Create a copy of the DataFrame to avoid modifying the original
        df_dropped = self.df.copy()
        columns_to_drop = [col for col in columns if col in df_dropped.columns]
        
        if columns_to_drop:
            # Save the dropped columns
            dropped_columns = df_dropped[columns_to_drop].copy()
            # Drop the columns
            df_dropped.drop(columns=columns_to_drop, inplace=True)
            print(f"Dropped columns: {columns_to_drop}")
        else:
            dropped_columns = pd.DataFrame()  # Return an empty DataFrame if no columns were dropped
            print("No columns to drop.")
        
        return df_dropped, dropped_columns

    def add_dropped_columns(self, df_dropped, dropped_columns):
        """
        Add previously dropped columns back to the cleaned DataFrame.

        Parameters:
        - df_dropped: The cleaned DataFrame with columns dropped.
        - dropped_columns: A DataFrame containing the previously dropped columns.

        Returns:
        - DataFrame: The DataFrame with the dropped columns added back.
        """
        if not dropped_columns.empty:
            df_combined = pd.concat([df_dropped, dropped_columns], axis=1)
            print("Added dropped columns back to the DataFrame.")
        else:
            df_combined = df_dropped
            print("No dropped columns to add back.")
        
        return df_combined

class HandsetAnalysis(DataFrameMissingValueChecker):
    def __init__(self, df):
        """
        Initialize the HandsetAnalysis class by inheriting the DataFrameMissingValueChecker.
        
        Parameters:
        - df: The pandas DataFrame to analyze.
        """
        super().__init__(df)

    def fill_missing_and_undefined(self):
        """
        Replace 'undefined' and NaN values in the 'Handset Type' and 'Handset Manufacturer' columns with 'Unknown'.
        """
        self.df['Handset Type'].replace('undefined', pd.NA, inplace=True)
        self.df['Handset Type'].fillna('Unknown', inplace=True)
        self.df['Handset Manufacturer'].replace('undefined', pd.NA, inplace=True)
        self.df['Handset Manufacturer'].fillna('Unknown', inplace=True)


    def fill_missing_numerical(self, method='mean', value=None):
        """
        Fill missing numerical data in all numerical columns using the chosen method.
        
        Args:
        method (str): The method to use for filling missing values. Options are 'mean', 'median', or 'value'.
        value (float/int): If method is 'value', this specifies the value to fill.
        """
        # Get all numerical columns
        numerical_columns = self.df.select_dtypes(include=[np.number]).columns
        
        for column_name in numerical_columns:
            if method == 'mean':
                fill_value = self.df[column_name].mean()
            elif method == 'median':
                fill_value = self.df[column_name].median()
            elif method == 'value' and value is not None:
                fill_value = value
            else:
                raise ValueError("Invalid method or missing value for 'value' method.")
            
            self.df[column_name].fillna(fill_value, inplace=True)
            print(f"Filled missing values in column '{column_name}' using method '{method}' with value {fill_value}.")
            
    def get_top_handsets(self):
        """
        Identify the top 10 handsets used by customers, excluding 'Unknown'.
        """
        self.fill_missing_and_undefined()
        if 'Handset Type' in self.df.columns:
            top_10_handsets = self.df[self.df['Handset Type'] != 'Unknown']['Handset Type'].value_counts().head(10)
            return top_10_handsets
        else:
            raise KeyError("Column 'Handset Type' not found in DataFrame.")

    def get_top_manufacturers(self):
        """
        Identify the top 3 handset manufacturers, excluding 'Unknown'.
        """
        if 'Handset Manufacturer' in self.df.columns:
            top_3_manufacturers = self.df[self.df['Handset Manufacturer'] != 'Unknown']['Handset Manufacturer'].value_counts().head(3)
            return top_3_manufacturers
        else:
            raise KeyError("Column 'Handset Manufacturer' not found in DataFrame.")

    def get_top_handsets_per_manufacturer(self):
        """
        Identify the top 5 handsets per top 3 handset manufacturers, excluding 'Unknown'.
        """
        top_3_manufacturers = self.get_top_manufacturers()
        top_5_per_manufacturer = {}
        for manufacturer in top_3_manufacturers.index:
            if 'Handset Type' in self.df.columns:
                top_5_handsets = self.df[(self.df['Handset Manufacturer'] == manufacturer) & 
                                         (self.df['Handset Type'] != 'Unknown')]['Handset Type'].value_counts().head(5)
                top_5_per_manufacturer[manufacturer] = top_5_handsets
            else:
                raise KeyError("Column 'Handset Type' not found in DataFrame.")
        return top_5_per_manufacturer