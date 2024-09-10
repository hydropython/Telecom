import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

class UserExperienceAnalyzer:
    def __init__(self, df):
        self.df = df

    def compute_tcp_stats(self):
        # Task 3.1a: Top, bottom, and most frequent TCP values (both DL and UL)
        tcp_columns = ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
        
        tcp_stats = {}
        for col in tcp_columns:
            top_10 = self.df[col].nlargest(10)
            bottom_10 = self.df[col].nsmallest(10)
            most_frequent = self.df[col].value_counts().head(10)
            tcp_stats[col] = {'Top 10': top_10, 'Bottom 10': bottom_10, 'Most Frequent': most_frequent}
            # Print numeric result
            print(f"\n{col} - Top 10 values:\n", top_10)
            print(f"\n{col} - Bottom 10 values:\n", bottom_10)
            print(f"\n{col} - Most Frequent values:\n", most_frequent)
        return tcp_stats

    def compute_rtt_stats(self):
        # Task 3.1b: Top, bottom, and most frequent RTT values (both DL and UL)
        rtt_columns = ['Avg RTT DL (ms)', 'Avg RTT UL (ms)']
        
        rtt_stats = {}
        for col in rtt_columns:
            top_10 = self.df[col].nlargest(10)
            bottom_10 = self.df[col].nsmallest(10)
            most_frequent = self.df[col].value_counts().head(10)
            rtt_stats[col] = {'Top 10': top_10, 'Bottom 10': bottom_10, 'Most Frequent': most_frequent}
            # Print numeric result
            print(f"\n{col} - Top 10 values:\n", top_10)
            print(f"\n{col} - Bottom 10 values:\n", bottom_10)
            print(f"\n{col} - Most Frequent values:\n", most_frequent)
        return rtt_stats

    def compute_throughput_stats(self):
        # Task 3.1c: Top, bottom, and most frequent throughput values (DL and UL)
        tp_columns = ['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']
        
        tp_stats = {}
        for col in tp_columns:
            top_10 = self.df[col].nlargest(10)
            bottom_10 = self.df[col].nsmallest(10)
            most_frequent = self.df[col].value_counts().head(10)
            tp_stats[col] = {'Top 10': top_10, 'Bottom 10': bottom_10, 'Most Frequent': most_frequent}
            # Print numeric result
            print(f"\n{col} - Top 10 values:\n", top_10)
            print(f"\n{col} - Bottom 10 values:\n", bottom_10)
            print(f"\n{col} - Most Frequent values:\n", most_frequent)
        return tp_stats

    def report_throughput_distribution(self):
        # Task 3.3: Distribution of average throughput per handset type
        throughput_columns = ['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']
        throughput_by_handset = self.df.groupby('Handset Type')[throughput_columns].mean()

        # Print numeric result
        print("\nAverage Throughput per Handset Type:\n", throughput_by_handset)

        # Plotting the distribution
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=throughput_by_handset, palette="Set3")
        plt.title('Distribution of Throughput by Handset Type')
        plt.ylabel('Throughput (kbps)')
        plt.xticks(rotation=90)
        plt.show()
        
        return throughput_by_handset

    def report_tcp_retransmission(self):
        # Task 3.3: Average TCP retransmission volume per handset type
        tcp_columns = ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
        tcp_by_handset = self.df.groupby('Handset Type')[tcp_columns].mean()

        # Print numeric result
        print("\nAverage TCP Retransmission per Handset Type:\n", tcp_by_handset)

        # Plotting the distribution
        plt.figure(figsize=(10, 6))
        tcp_by_handset.plot(kind='bar', stacked=True)
        plt.title('Average TCP Retransmission by Handset Type')
        plt.ylabel('TCP Retransmission Volume (Bytes)')
        plt.xticks(rotation=90)
        plt.show()
        
        return tcp_by_handset

    def perform_kmeans_clustering(self, n_clusters=3):
        # Task 3.4: K-means clustering based on experience metrics
        features = ['Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)', 
                    'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 
                    'Avg RTT DL (ms)', 'Avg RTT UL (ms)']

        X = self.df[features].dropna()  # Dropping rows with missing values
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['Cluster'] = kmeans.fit_predict(X)

        # Print numeric result
        cluster_summary = self.df.groupby('Cluster')[features].mean()
        print("\nCluster Summary:\n", cluster_summary)

        return cluster_summary
    
    
    