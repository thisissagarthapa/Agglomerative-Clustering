# Football Players Performance Clustering

This project applies Agglomerative Clustering to cluster football players based on their performance metrics such as Goals, Assists, Tackles, and Pass Accuracy.

## Project Overview

The goal of this project is to perform hierarchical clustering on football players' data using the Agglomerative Clustering algorithm. The relationships between performance metrics are visualized using a dendrogram, and the clusters are displayed using Seaborn's `pairplot`.

## Dataset

The dataset used in this project contains the following features:

| Player Name       | Position    | Goals | Assists | Tackles | Pass Accuracy | Dribbles per Game | Interceptions | Shots per Game | Yellow Cards | Red Cards |
|-------------------|-------------|-------|---------|---------|---------------|-------------------|---------------|----------------|--------------|-----------|
| Neymar            | Midfielder  | 17    | 10      | 103     | 84.20         | 4.11              | 0             | 4.16           | 3            | 1         |
| Neymar            | Midfielder  | 3     | 8       | 52      | 96.36         | 5.18              | 26            | 0.85           | 0            | 1         |
| Kevin De Bruyne   | Defender    | 25    | 21      | 12      | 83.16         | 5.52              | 4             | 1.38           | 0            | 2         |
| Kevin De Bruyne   | Forward     | 23    | 13      | 118     | 97.83         | 3.78              | 49            | 0.18           | 0            | 0         |
| Kevin De Bruyne   | Midfielder  | 24    | 6       | 47      | 88.44         | 0.62              | 4             | 1.15           | 9            | 1         |

We focus on the following features for clustering:

- `Goals`
- `Assists`
- `Tackles`
- `Pass Accuracy`

## Libraries Used

- `numpy`
- `pandas`
- `seaborn`
- `matplotlib`
- `scipy`
- `sklearn`

## Clustering Steps

1. **Data Preparation**: 
   We extracted the relevant columns (`Goals`, `Assists`, `Tackles`, `Pass Accuracy`) from the dataset for clustering.

2. **Pairplot Visualization**: 
   Visualized relationships between these features using Seaborn's `pairplot`.

3. **Hierarchical Clustering**: 
   - Generated a dendrogram using `scipy` to visualize the hierarchical relationships between clusters.
   - Performed Agglomerative Clustering with 4 clusters and linkage method as 'single'.

4. **Clustering Results**: 
   The clusters were added to the dataset, and the results were visualized using Seaborn's `pairplot`, with clusters indicated by different colors.


 ## Screenshots
 
![Screenshot (140)](https://github.com/user-attachments/assets/88ba1aed-01e4-428f-a913-dcafe6b76c1d)

![Screenshot (141)](https://github.com/user-attachments/assets/b08eeca9-9ed8-4d52-8cd3-798de2c0221b)

![Screenshot (142)](https://github.com/user-attachments/assets/e7c202b0-ff25-4e12-9f3b-e2706a46264a)


## Code Example

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
datasets = pd.read_csv("football_data.csv")

# Select relevant features for clustering
new_dataset = datasets[["Goals", "Assists", "Tackles", "Pass Accuracy"]]

# Pairplot to visualize relationships
sns.pairplot(data=new_dataset)
plt.show()

# Hierarchical clustering - Dendrogram
from scipy.cluster import hierarchy as sc
sc.dendrogram(sc.linkage(new_dataset, method='single', metric='euclidean'))
plt.show()

# Perform Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering
ag = AgglomerativeClustering(n_clusters=4, linkage='single')
new_dataset["predicts"] = ag.fit_predict(new_dataset)

# Visualize clusters
sns.pairplot(new_dataset, hue="predicts")
plt.show()


