import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class UserBehaviorAnalysis:
    def __init__(self, df, user_id_col, session_duration_col=None):
        self.df = df
        self.user_id_col = user_id_col
        self.session_duration_col = session_duration_col
        self.dl_columns = ['Social Media DL (Bytes)',  'Netflix DL (Bytes)', 
                   'Google DL (Bytes)', 'Email DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
        self.ul_columns = ['Social Media UL (Bytes)', 'Netflix UL (Bytes)', 
                   'Google UL (Bytes)', 'Email UL (Bytes)', 'Gaming UL (Bytes)', 'Other UL (Bytes)']

        self._prepare_data()

    def _prepare_data(self):
        self.df['Total DL (Bytes)'] = self.df[self.dl_columns].sum(axis=1)
        self.df['Total UL (Bytes)'] = self.df[self.ul_columns].sum(axis=1)
        self.df['Total Data (Bytes)'] = self.df['Total DL (Bytes)'] + self.df['Total UL (Bytes)']
        self.user_aggregated = self.df.groupby(self.user_id_col).agg(
            number_of_sessions=('MSISDN/Number', 'size'),
            total_dl=('Total DL (Bytes)', 'sum'),
            total_ul=('Total UL (Bytes)', 'sum'),
            total_data=('Total Data (Bytes)', 'sum'),
            session_duration=('Dur. (ms)', 'sum') if self.session_duration_col else 'size'
        ).reset_index()
        
        # Define custom bin edges
        # Define custom bin edges
        bin_edges = [0, 8125, 48151, 86399, 138740, 3846645]

        # Create the binned column
        self.user_aggregated['decile_class'] = pd.cut(self.user_aggregated['session_duration'], bins=bin_edges, labels=False, include_lowest=True) + 1

    def describe_variables(self):
        description = self.df.describe(include='all')
        print("Variables and their data types:\n", description.dtypes)
        print("\nSummary Statistics:\n", description)
        return description

    def transform_and_aggregate(self):
        decile_aggregation = self.user_aggregated.groupby('decile_class').agg(
            total_dl=('total_dl', 'sum'),
            total_ul=('total_ul', 'sum'),
            total_data=('total_data', 'sum')
        )
        print("\nDecile Class Aggregation:\n", decile_aggregation)
        return decile_aggregation

    def univariate_analysis(self):
        quantitative_vars = self.df.select_dtypes(include=[np.number]).columns
        dispersion = self.df[quantitative_vars].agg([np.mean, np.median, np.std, np.var])
        print("\nDispersion Parameters:\n", dispersion)
        return dispersion

    def graphical_univariate_analysis(self):
        quantitative_vars = self.df.select_dtypes(include=[np.number]).columns
        for var in quantitative_vars:
            plt.figure(figsize=(12, 6))
            sns.histplot(self.df[var], kde=True)
            plt.title(f'Histogram and KDE for {var}')
            plt.xlabel(var)
            plt.ylabel('Frequency')
            plt.show()

    def bivariate_analysis(self):
        for app in ['Social Media', 'YouTube', 'Netflix', 'Google', 'Email', 'Gaming', 'Other']:
            app_dl_col = f'{app} DL (Bytes)'
            app_ul_col = f'{app} UL (Bytes)'
            self.df[f'{app}_Total'] = self.df[app_dl_col] + self.df[app_ul_col]
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=self.df[f'{app}_Total'], y=self.df['Total Data (Bytes)'])
            plt.title(f'Relationship between {app} Total Data and Total Data')
            plt.xlabel(f'{app} Total Data (Bytes)')
            plt.ylabel('Total Data (Bytes)')
            plt.show()

    def correlation_analysis(self):
            # List of expected columns
            dl_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                        'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
            
            # Ensure columns are present in DataFrame
            existing_columns = [col for col in dl_columns if col in self.df.columns]
            
            # Calculate correlation matrix
            correlation_matrix = self.df[existing_columns].corr()
            
            # Print the correlation matrix
            print("Correlation Matrix:\n", correlation_matrix)
            
            # Plot the heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
            plt.title('Correlation Matrix of Data Usage')
            plt.show()

    def dimensionality_reduction(self):
        # Standardize the data
        features = self.df[self.dl_columns + self.ul_columns]
        features_standardized = StandardScaler().fit_transform(features)
        
        # Apply PCA
        pca = PCA(n_components=min(features_standardized.shape))
        principal_components = pca.fit_transform(features_standardized)
        explained_variance = pca.explained_variance_ratio_
        
        print("\nPrincipal Component Analysis Results:")
        print(f"Explained Variance Ratio: {explained_variance}")
        print(f"Explained Variance Ratio Cumulative Sum: {np.cumsum(explained_variance)}")
        
        return principal_components, explained_variance




