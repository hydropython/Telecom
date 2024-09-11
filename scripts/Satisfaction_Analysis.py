import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

class CustomerSatisfactionAnalyzer:
    def __init__(self, df):
        self.data = df

    def calculate_scores(self):
        # Example of creating 'satisfaction_score', 'engagement_score', and 'experience_score' columns
        # Replace these calculations with your actual logic
        self.data['satisfaction_score'] = (self.data['Avg RTT DL (ms)'] + self.data['Avg RTT UL (ms)']) / 2
        self.data['engagement_score'] = self.data['HTTP DL (Bytes)'] / self.data['Total DL (Bytes)'] * 100
        self.data['experience_score'] = self.data['Total UL (Bytes)'] / self.data['Total DL (Bytes)'] * 100
        
        # Handle missing values by filling them with the column mean for numeric columns only
        self.data.fillna(self.data.mean(numeric_only=True), inplace=True)

        return self.data

    def regression_model(self):
        # Ensure required columns are present
        required_columns = ['engagement_score', 'experience_score', 'satisfaction_score']
        for col in required_columns:
            if col not in self.data.columns:
                raise KeyError(f"Column '{col}' is missing from the DataFrame")
        
        # Build a regression model to predict satisfaction score
        X = self.data[['engagement_score', 'experience_score']]
        y = self.data['satisfaction_score']
        model = LinearRegression()
        model.fit(X, y)
        print("Regression model coefficients:", model.coef_)
        print("Regression model intercept:", model.intercept_)

        return model

    def show_top_customers(self):
        # Ensure 'satisfaction_score' column is present
        if 'satisfaction_score' not in self.data.columns:
            raise KeyError("'satisfaction_score' column is missing from the DataFrame")

        # Display top 10 satisfied customers
        top_customers = self.data.nsmallest(10, 'satisfaction_score')  # Smaller score means better satisfaction
        print("Top 10 Satisfied Customers:")
        print(top_customers[['MSISDN/Number', 'satisfaction_score']])

        # Plot top 10 customers with modern color palette
        plt.figure(figsize=(12, 8))
        sns.barplot(x='satisfaction_score', y='MSISDN/Number', data=top_customers, palette='coolwarm')
        plt.xlabel('Satisfaction Score')
        plt.title('Top 10 Satisfied Customers')
        plt.gca().invert_yaxis()  # Invert y-axis to have the highest score on top
        plt.show()

    def cluster_customers(self):
        # Perform KMeans clustering on satisfaction scores
        kmeans = KMeans(n_clusters=2, random_state=42)
        self.data['satisfaction_cluster'] = kmeans.fit_predict(self.data[['engagement_score', 'experience_score']])
        print("Clusters assigned to customers based on engagement and experience scores.")

    def aggregate_cluster_scores(self):
        # Aggregate and display average satisfaction and experience scores per cluster
        cluster_avg = self.data.groupby('satisfaction_cluster')[['satisfaction_score', 'experience_score']].mean()
        print("Average Satisfaction and Experience Score per Cluster:")
        print(cluster_avg)

    def visualize_data(self):
        # Visualize engagement vs experience with clusters
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(self.data['engagement_score'], self.data['experience_score'], 
                              c=self.data['satisfaction_cluster'], cmap='viridis', s=100)
        plt.title('Customer Engagement vs Experience (Clustered)')
        plt.xlabel('Engagement Score')
        plt.ylabel('Experience Score')
        plt.colorbar(scatter, label='Cluster')
        plt.show()