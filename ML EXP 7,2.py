import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

df = pd.read_csv(r"C:\Users\SHALINI A\Downloads\archive (21).zip")
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
gmm = GaussianMixture(n_components=5, random_state=42)
gmm.fit(X_scaled)
clusters = gmm.predict(X_scaled)
df['GMM_Cluster'] = clusters
print("Silhouette Score:", silhouette_score(X_scaled, clusters))
print("Log Likelihood:", gmm.score(X_scaled))
cluster_summary = df.groupby('GMM_Cluster').mean(numeric_only=True)
print("\nCluster Summary:")
print(cluster_summary)
probabilities = gmm.predict_proba(X_scaled)
plt.hist(probabilities.max(axis=1), bins=20)
plt.title("Cluster Probability Distribution")
plt.xlabel("Max Probability")
plt.ylabel("Number of Points")
plt.show()

x = np.linspace(X_scaled[:, 0].min(), X_scaled[:, 0].max(), 100)
y = np.linspace(X_scaled[:, 1].min(), X_scaled[:, 1].max(), 100)

X_grid, Y_grid = np.meshgrid(x, y)
grid = np.array([X_grid.ravel(), Y_grid.ravel()]).T

Z = -gmm.score_samples(grid)
Z = Z.reshape(X_grid.shape)

plt.contour(X_grid, Y_grid, Z)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)

plt.title("GMM Contour Plot")
plt.xlabel("Income")
plt.ylabel("Spending")
plt.show()

kmeans = KMeans(n_clusters=5, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_clusters)
plt.title("K-Means Clustering")

plt.subplot(1,2,2)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters)
plt.title("GMM Clustering")
plt.show()

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
