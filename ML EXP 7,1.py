print("24BAD409_SHALINI A")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
df = pd.read_csv(r"C:\Users\SHALINI A\Downloads\archive (21).zip")

X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
inertia = []
K_range = range(1, 11)
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X_scaled)
    inertia.append(model.inertia_)

plt.plot(K_range, inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.show()
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters
print("Silhouette Score:", silhouette_score(X_scaled, clusters))
print(df.groupby('Cluster').mean(numeric_only=True))
for i in range(5):
    plt.scatter(X_scaled[clusters == i, 0],
                X_scaled[clusters == i, 1],
                label=f'Cluster {i}')

plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200, marker='X', label='Centroids')

plt.title("K-Means Clustering")
plt.xlabel("Income")
plt.ylabel("Spending")
plt.legend()
plt.show()
cluster_summary = df.groupby('Cluster').mean(numeric_only=True)
print("\nCluster Summary:")
print(cluster_summary)
print("\nCluster Interpretation:")
for i in cluster_summary.index:
    income = cluster_summary.loc[i, 'Annual Income (k$)']
    spending = cluster_summary.loc[i, 'Spending Score (1-100)']
    
    if income > 70 and spending > 70:
        category = "High Income - High Spending (Target Customers)"
    elif income > 70 and spending < 40:
        category = "High Income - Low Spending (Potential Customers)"
    elif income < 40 and spending > 70:
        category = "Low Income - High Spending (Risky Customers)"
    elif income < 40 and spending < 40:
        category = "Low Income - Low Spending (Least Valuable)"
    else:
        category = "Medium Income - Medium Spending (Average Customers)"
    print(f"Cluster {i}: {category}")
