import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class LuxuriousPlotter:
    """
    Class for creating luxurious visualizations with custom color schemes.
    """
    
    def __init__(self, handset_df, handset_manufacturer):
        """
        Initialize the LuxuriousPlotter with handset type and manufacturer data.
        
        Parameters:
        - handset_df: A pandas Series containing 'Handset Type'.
        - handset_manufacturer: A pandas Series containing 'Handset Manufacturer'.
        """
        self.handset_df = handset_df
        self.handset_manufacturer = handset_manufacturer

    def plot_top_handsets(self):
        """
        Plot the top 10 handsets with luxurious colors.
        """
        top_10_handsets = self.handset_df.value_counts().head(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_10_handsets.values, y=top_10_handsets.index, palette='YlGnBu')
        plt.title('Top 10 Handsets Used by Customers', fontsize=18, color='#004d4d')
        plt.xlabel('Number of Handsets', fontsize=14, color='#006666')
        plt.ylabel('Handset Type', fontsize=14, color='#006666')
        plt.grid(True, which='both', axis='x', linestyle='--', alpha=0.7)
        plt.show()

    def plot_top_manufacturers(self):
        """
        Plot the top 3 handset manufacturers with luxurious colors.
        """
        top_3_manufacturers = self.handset_manufacturer.value_counts().head(3)
        plt.figure(figsize=(8, 5))
        sns.barplot(x=top_3_manufacturers.index, y=top_3_manufacturers.values, palette='BuPu')
        plt.title('Top 3 Handset Manufacturers', fontsize=18, color='#2b004b')
        plt.xlabel('Manufacturer', fontsize=14, color='#660066')
        plt.ylabel('Number of Handsets', fontsize=14, color='#660066')
        plt.grid(True, which='both', axis='y', linestyle='--', alpha=0.7)
        plt.show()

    def plot_top_handsets_per_manufacturer(self):
        """
        Plot the top 5 handsets for each of the top 3 manufacturers with luxurious colors.
        """
        top_3_manufacturers = self.handset_manufacturer.value_counts().head(3)
        top_5_per_manufacturer = {}

        for manufacturer in top_3_manufacturers.index:
            top_5_handsets = self.handset_df[self.handset_manufacturer == manufacturer].value_counts().head(5)
            top_5_per_manufacturer[manufacturer] = top_5_handsets

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        colors = ['#8e44ad', '#3498db', '#e67e22']

        for i, (manufacturer, handsets) in enumerate(top_5_per_manufacturer.items()):
            sns.barplot(x=handsets.values, y=handsets.index, palette='GnBu', ax=axes[i])
            axes[i].set_title(f'Top 5 Handsets for {manufacturer}', fontsize=16, color=colors[i])
            axes[i].set_xlabel('Number of Handsets', fontsize=12, color=colors[i])
            axes[i].set_ylabel('Handset Type', fontsize=12, color=colors[i])
            axes[i].grid(True, which='both', axis='x', linestyle='--', alpha=0.7)

        plt.tight_layout()
        plt.show()