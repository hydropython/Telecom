import numpy as np
import pandas as pd
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import plotly.express as px
import plotly.graph_objects as go

class CustomerSatisfactionAnalysis:
    def __init__(self, user_data, less_engaged_centroid, worst_experience_centroid):
        self.user_data = user_data
        self.less_engaged_centroid = less_engaged_centroid
        self.worst_experience_centroid = worst_experience_centroid
        self.regression_model = None
    
    def calculate_euclidean_distance(self, user_point, cluster_centroid):
        return np.linalg.norm(user_point - cluster_centroid)

    def assign_scores(self):
        """Assign engagement and experience scores based on Euclidean distance."""
        self.user_data['engagement_score'] = self.user_data.apply(
            lambda row: self.calculate_euclidean_distance(row[['Engagement_Metric1', 'Engagement_Metric2']], self.less_engaged_centroid), axis=1
        )
        self.user_data['experience_score'] = self.user_data.apply(
            lambda row: self.calculate_euclidean_distance(row[['Experience_Metric1', 'Experience_Metric2']], self.worst_experience_centroid), axis=1
        )
    
    def calculate_satisfaction_score(self):
        """Calculate the average of engagement and experience scores as satisfaction score."""
        self.user_data['satisfaction_score'] = self.user_data[['engagement_score', 'experience_score']].mean(axis=1)
        top_10_satisfied_customers = self.user_data.sort_values(by='satisfaction_score', ascending=False).head(10)
        return top_10_satisfied_customers

    def build_regression_model(self):
        """Build and train a regression model to predict satisfaction scores."""
        X = self.user_data[['engagement_score', 'experience_score']]
        y = self.user_data['satisfaction_score']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.regression_model = LinearRegression()
        self.regression_model.fit(X_train, y_train)
        y_pred = self.regression_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"Mean Squared Error: {mse}")

    def run_kmeans_clustering(self):
        """Run K-means clustering (k=2) on engagement and experience scores."""
        kmeans = KMeans(n_clusters=2, random_state=42)
        self.user_data['cluster'] = kmeans.fit_predict(self.user_data[['engagement_score', 'experience_score']])
        return self.user_data.groupby('cluster').agg({'satisfaction_score': 'mean', 'experience_score': 'mean'}).reset_index()

    def visualize_scores(self):
        """Generate interesting plots for engagement, experience, and satisfaction scores."""
        fig = px.scatter(self.user_data, x='engagement_score', y='experience_score', color='satisfaction_score',
                         title="Engagement vs Experience Scores Colored by Satisfaction", hover_data=['user_id'])
        fig.update_layout(template='plotly_dark')
        fig.show()

    def visualize_kmeans(self):
        """Plot K-means clusters."""
        fig = px.scatter(self.user_data, x='engagement_score', y='experience_score', color='cluster',
                         title="K-Means Clustering on Engagement and Experience Scores")
        fig.update_layout(template='plotly_dark')
        fig.show()

    def plot_top_10_customers(self):
        """Visualize top 10 satisfied customers."""
        top_10 = self.calculate_satisfaction_score()
        fig = px.bar(top_10, x='user_id', y='satisfaction_score', title="Top 10 Satisfied Customers")
        fig.update_layout(template='plotly_dark')
        fig.show()

# Example usage
less_engaged_centroid = np.array([2.5, 3.0])  # Replace with actual centroid
worst_experience_centroid = np.array([1.0, 2.0])  # Replace with actual centroid

user_data = pd.DataFrame({
    'user_id': [1, 2, 3],  # User IDs
    'Engagement_Metric1': [1.0, 2.0, 3.0],
    'Engagement_Metric2': [2.5, 2.0, 3.5],
    'Experience_Metric1': [1.5, 3.0, 2.0],
    'Experience_Metric2': [2.0, 3.5, 1.5]
})

analysis = CustomerSatisfactionAnalysis(user_data, less_engaged_centroid, worst_experience_centroid)
analysis.assign_scores()  # Assign scores
analysis.build_regression_model()  # Build regression model
analysis.run_kmeans_clustering()  # Perform K-means clustering
analysis.visualize_scores()  # Visualize engagement and experience scores
analysis.visualize_kmeans()  # Visualize K-means clusters
analysis.plot_top_10_customers()  # Visualize top 10 customers