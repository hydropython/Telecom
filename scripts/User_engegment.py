import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class UserEngagementAnalysis:
    def __init__(self, df):
        self.df = df
        self.scaler = StandardScaler()

    def aggregate_metrics(self):
        # Aggregate metrics per customer
        self.agg_df = self.df.groupby('MSISDN/Number').agg(
            session_frequency=('Bearer Id', 'count'),
            total_duration=('Dur. (ms)', 'sum'),
            total_traffic_dl=('Total DL (Bytes)', 'sum'),
            total_traffic_ul=('Total UL (Bytes)', 'sum')
        ).reset_index()
        print("Aggregated Metrics:")
        print(self.agg_df.head())
        return self.agg_df

    def normalize_metrics(self):
        # Normalize the metrics
        self.agg_df[['session_frequency', 'total_duration', 'total_traffic_dl', 'total_traffic_ul']] = self.scaler.fit_transform(
            self.agg_df[['session_frequency', 'total_duration', 'total_traffic_dl', 'total_traffic_ul']]
        )
        print("Normalized Metrics:")
        print(self.agg_df.head())
        return self.agg_df

    def kmeans_clustering(self, k=3):
        # Perform k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=0)
        self.agg_df['cluster'] = kmeans.fit_predict(
            self.agg_df[['session_frequency', 'total_duration', 'total_traffic_dl', 'total_traffic_ul']]
        )
        print("Cluster Assignments:")
        print(self.agg_df[['MSISDN/Number', 'cluster']].head())
        return self.agg_df['cluster'].value_counts()

    def compute_cluster_stats(self):
        # Compute stats for each cluster
        self.cluster_stats = self.agg_df.groupby('cluster').agg(
            min_frequency=('session_frequency', 'min'),
            max_frequency=('session_frequency', 'max'),
            avg_frequency=('session_frequency', 'mean'),
            total_frequency=('session_frequency', 'sum'),
            min_duration=('total_duration', 'min'),
            max_duration=('total_duration', 'max'),
            avg_duration=('total_duration', 'mean'),
            total_duration=('total_duration', 'sum'),
            min_traffic_dl=('total_traffic_dl', 'min'),
            max_traffic_dl=('total_traffic_dl', 'max'),
            avg_traffic_dl=('total_traffic_dl', 'mean'),
            total_traffic_dl=('total_traffic_dl', 'sum'),
            min_traffic_ul=('total_traffic_ul', 'min'),
            max_traffic_ul=('total_traffic_ul', 'max'),
            avg_traffic_ul=('total_traffic_ul', 'mean'),
            total_traffic_ul=('total_traffic_ul', 'sum')
        ).reset_index()
        print("Cluster Statistics:")
        print(self.cluster_stats)
        return self.cluster_stats

    def plot_engagement_metrics(self):
        # Plot top 10 customers per engagement metric
        fig, axes = plt.subplots(3, 1, figsize=(14, 18))
        top_10_freq = self.agg_df.nlargest(10, 'session_frequency')
        top_10_duration = self.agg_df.nlargest(10, 'total_duration')
        top_10_traffic_dl = self.agg_df.nlargest(10, 'total_traffic_dl')
        top_10_traffic_ul = self.agg_df.nlargest(10, 'total_traffic_ul')

        sns.barplot(x='session_frequency', y='MSISDN/Number', data=top_10_freq, palette="viridis", ax=axes[0])
        axes[0].set_title('Top 10 Customers by Session Frequency')

        sns.barplot(x='total_duration', y='MSISDN/Number', data=top_10_duration, palette="viridis", ax=axes[1])
        axes[1].set_title('Top 10 Customers by Total Duration')

        sns.barplot(x='total_traffic_dl', y='MSISDN/Number', data=top_10_traffic_dl, palette="viridis", ax=axes[2])
        axes[2].set_title('Top 10 Customers by Total Download Traffic')

        plt.tight_layout()
        plt.show()

    def plot_application_usage(self):
        app_usage = {}
        app_columns = {
            'Social Media': ['Social Media DL (Bytes)', 'Social Media UL (Bytes)'],
            'YouTube': ['YouTube DL (Bytes)', 'YouTube UL (Bytes)'],
            'Netflix': ['Netflix DL (Bytes)', 'Netflix UL (Bytes)'],
            'Google': ['Google DL (Bytes)', 'Google UL (Bytes)'],
            'Email': ['Email DL (Bytes)', 'Email UL (Bytes)'],
            'Gaming': ['Gaming DL (Bytes)', 'Gaming UL (Bytes)'],
            'Other': ['Other DL (Bytes)', 'Other UL (Bytes)']
        }

        for app, cols in app_columns.items():
            try:
                app_usage[app] = self.df[cols].sum().sum()
            except KeyError as e:
                print(f"Column error: {e}")

        top_apps = sorted(app_usage.items(), key=lambda x: x[1], reverse=True)[:3]
        apps, usage = zip(*top_apps)

        plt.figure(figsize=(12, 6))
        sns.barplot(x=apps, y=usage, palette="viridis")
        plt.title('Top 3 Most Used Applications')
        plt.xlabel('Application')
        plt.ylabel('Total Data Volume (Bytes)')
        plt.show()
        
        return top_apps

    def plot_elbow_method(self):
        # Determine the optimal k using the elbow method
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, random_state=0)
            kmeans.fit(self.agg_df[['session_frequency', 'total_duration', 'total_traffic_dl', 'total_traffic_ul']])
            wcss.append(kmeans.inertia_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, 11), wcss, marker='o', color='purple')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of Clusters')
        plt.ylabel('WCSS')
        plt.grid(True)
        plt.show()

        return wcss
