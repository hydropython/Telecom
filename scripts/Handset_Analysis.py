import pandas as pd

class HandsetAnalysis:
    def __init__(self, df):
        self.df = df

    def fill_missing_and_undefined(self):
        """
        Replace 'undefined' and NaN values in the 'Handset Type' and 'Handset Manufacturer' columns with 'Unknown'.
        """
        self.df['Handset Type'].replace('undefined', pd.NA, inplace=True)
        self.df['Handset Type'].fillna('Unknown', inplace=True)
        self.df['Handset Manufacturer'].replace('undefined', pd.NA, inplace=True)
        self.df['Handset Manufacturer'].fillna('Unknown', inplace=True)

    def get_top_handsets(self):
        """
        Identify the top 10 handsets used by customers.
        """
        self.fill_missing_and_undefined()
        if 'Handset Type' in self.df.columns:
            top_10_handsets = self.df['Handset Type'].value_counts().head(10)
            return top_10_handsets
        else:
            raise KeyError("Column 'Handset Type' not found in DataFrame.")

    def get_top_manufacturers(self):
        """
        Identify the top 3 handset manufacturers.
        """
        if 'Handset Manufacturer' in self.df.columns:
            top_3_manufacturers = self.df['Handset Manufacturer'].value_counts().head(3)
            return top_3_manufacturers
        else:
            raise KeyError("Column 'Handset Manufacturer' not found in DataFrame.")

    def get_top_handsets_per_manufacturer(self):
        """
        Identify the top 5 handsets per top 3 handset manufacturers.
        """
        top_3_manufacturers = self.get_top_manufacturers()
        top_5_per_manufacturer = {}
        for manufacturer in top_3_manufacturers.index:
            if 'Handset Type' in self.df.columns:
                top_5_handsets = self.df[self.df['Handset Manufacturer'] == manufacturer]['Handset Type'].value_counts().head(5)
                top_5_per_manufacturer[manufacturer] = top_5_handsets
            else:
                raise KeyError("Column 'Handset Type' not found in DataFrame.")
        return top_5_per_manufacturer