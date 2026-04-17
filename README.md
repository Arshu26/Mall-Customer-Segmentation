# Mall-Customer-Segmentation

Step 1️⃣ Import Libraries

Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

Step 2️⃣ Load Dataset

Load dataset
data = pd.read_csv("Mall_Customers.csv")

Display first 5 rows
data.head()

Step 3️⃣ Check Dataset Information

Dataset information
data.info()

Statistical summary
data.describe()

Step 4️⃣ Select Features for Clustering

Selecting Annual Income and Spending Score
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

Step 5️⃣ Use Elbow Method to Find Optimal Clusters

Finding optimal number of clusters
wcss = []

for i in range(1, 11):
kmeans = KMeans(n_clusters=i, random_state=42)
kmeans.fit(X)
wcss.append(kmeans.inertia_)

Plot Elbow Graph
plt.plot(range(1, 11), wcss)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

Step 6️⃣ Apply K-Means Algorithm

Apply K-Means with 5 clusters
kmeans = KMeans(n_clusters=5, random_state=42)

Predict clusters
y_kmeans = kmeans.fit_predict(X)

Add cluster column to dataset
data['Cluster'] = y_kmeans

Step 7️⃣ Visualize Clusters

Visualizing clusters
plt.scatter(
X.iloc[:, 0],
X.iloc[:, 1],
c=y_kmeans
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

Step 8️⃣ Display Cluster Centers

Show cluster centers
print("Cluster Centers:")
print(kmeans.cluster_centers_)
