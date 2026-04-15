 K-MEANS CLUSTERING

 Dataset Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

This project focuses on customer segmentation using the K-Means clustering algorithm on the Mall Customer Segmentation dataset from Kaggle. The main objective is to group customers based on similarities in Annual Income and Spending Score to understand different customer behaviors. K-Means is an unsupervised learning algorithm that partitions data into a fixed number of clusters by minimizing the distance between data points and their respective cluster centroids.

Before applying the algorithm, the dataset is preprocessed by selecting relevant features and ensuring data quality. The optimal number of clusters is identified using the Elbow Method. The final clusters are visualized using scatter plots, where each group represents a distinct category of customers. K-Means is simple, fast, and effective for well-separated clusters, but it assumes spherical cluster shapes and is sensitive to initialization.

 GAUSSIAN MIXTURE MODEL (GMM)

 Dataset Link: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

This project implements customer segmentation using the Gaussian Mixture Model (GMM), a probabilistic clustering technique. The goal is to group customers while capturing the uncertainty and overlap between clusters using the same dataset. GMM assumes that data points are generated from a mixture of Gaussian distributions and assigns probabilities to each point for belonging to different clusters.

The dataset is preprocessed similarly by selecting key features such as Annual Income and Spending Score. After fitting the model, the clustering results are visualized, showing soft cluster assignments. GMM is more flexible than K-Means as it can handle overlapping and non-spherical clusters. However, it is computationally more complex and requires proper parameter tuning for accurate results.

# 24ADI003_24BAD409_EXP_7
