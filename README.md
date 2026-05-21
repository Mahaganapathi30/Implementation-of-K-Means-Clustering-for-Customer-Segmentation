# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the customer dataset and select relevant features such as Annual Income and Spending Score

2. Choose the number of clusters K and initialize K centroids randomly

3. Assign each data point to the nearest centroid using Euclidean distance and update centroids by computing the mean of each cluster

4. Repeat the assignment and update steps until centroids stop changing and display the final customer clusters

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: MAHAGANAPATHI R
RegisterNumber: 212225040221
*/

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("Mall_Customer.csv")
X = data.iloc[:, [3, 4]].values
kmeans = KMeans(n_clusters=5, random_state=0)
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50)
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            s=200,
            marker='X')
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.title("Customer Segmentation using K-Means")
plt.show()


```

## Output:

<img width="1919" height="916" alt="image" src="https://github.com/user-attachments/assets/1832bb20-6370-49ed-80b8-32ea777c1f18" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
