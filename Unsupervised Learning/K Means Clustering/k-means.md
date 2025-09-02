# K-Means Clustering

## Introduction

It works by iteratively assigning data points to the nearest cluster centroid and then updating the centroids based on the mean of all points in each cluster.

![K-Means Clustering Visualization](/images/k-means.png)

---

## Fundamental Concepts

### Basic Idea

K-Means aims to partition n observations into k clusters where each observation belongs to the cluster with the nearest mean (centroid), serving as a prototype of the cluster.

### Key Components

- **Centroids**: The center points of each cluster, calculated as the mean of all points in that cluster
- **Clusters**: Groups of data points that are closer to their centroid than to any other centroid
- **Distance Metric**: Usually Euclidean distance, though other metrics can be used

---

## Mathematical Foundation

### Objective Function

K-Means minimizes the Within-Cluster Sum of Squares (WCSS):

$$WCSS = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2$$

Where:

- $k$ is the number of clusters
- $C_i$ is the $i^{th}$ cluster
- $\mu_i$ is the centroid of cluster $i$
- $x$ represents data points in cluster $C_i$

### Distance Calculation

For two points $p = (p_1, p_2, ..., p_n)$ and $q = (q_1, q_2, ..., q_n)$ in n-dimensional space:

**Euclidean Distance**:
$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

**Manhattan Distance**:
$$d(p, q) = \sum_{i=1}^{n} |p_i - q_i|$$

---

## Algorithm Steps

### Step 1: Initialize Centroids

1. **Random Initialization**: Choose k random points from the dataset as initial centroids
2. **K-Means++**: More sophisticated initialization that spreads initial centroids apart
3. **Manual Initialization**: Specify initial centroids based on domain knowledge

### Step 2: Assignment Phase

For each data point $x_i$:

- Calculate distance to all k centroids
- Assign $x_i$ to the cluster with the nearest centroid

$$C_i = \{x : \|x - \mu_i\| \leq \|x - \mu_j\| \text{ for all } j \neq i\}$$

### Step 3: Update Phase

Recalculate centroids as the mean of all points in each cluster:

$$\mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x$$

Where $|C_i|$ is the number of points in cluster $C_i$.

### Step 4: Convergence

- Repeat Steps 2 and 3 until:
  - Centroids no longer move significantly
  - Maximum number of iterations reached
  - WCSS improvement falls below threshold

---

## How to Select the Optimal Value of k

### Elbow Method

The most popular method for determining the optimal number of clusters:

1. **Calculate WCSS** for different values of k (1, 2, 3, ..., max_k)
2. **Plot WCSS vs k**
3. **Look for the "elbow"** - the point where the rate of decrease slows down significantly

**WCSS Formula**:
$$WCSS = \sum_{i=1}^{n} (distance\ of\ point_i\ from\ centroid)^2$$

**Interpretation**:

- **k=1**: WCSS is highest (all points in one cluster)
- **k=2**: WCSS reduces significantly
- **k=3, 4, ...**: WCSS continues to decrease but at a slower rate
- **Optimal k**: The point where adding more clusters doesn't provide substantial improvement

### Other Methods

1. **Silhouette Analysis**: Measures how similar an object is to its own cluster compared to other clusters
2. **Gap Statistic**: Compares the total within-cluster variation with expected values under null reference distribution
3. **Calinski-Harabasz Index**: Ratio of between-cluster dispersion and within-cluster dispersion
4. **Davies-Bouldin Index**: Average similarity measure of each cluster with its most similar cluster

---

## Implementation Considerations

### Hyperparameters

- **n_clusters (k)**: Number of clusters to form
- **max_iter**: Maximum number of iterations for a single run
- **n_init**: Number of times the algorithm will be run with different centroid seeds
- **init**: Method for initialization ('k-means++', 'random', or array-like)
- **tol**: Relative tolerance with regards to inertia to declare convergence

### Data Preprocessing

1. **Feature Scaling**: Essential since K-Means is distance-based
   - StandardScaler: $(x - \mu) / \sigma$
   - MinMaxScaler: $(x - x_{min}) / (x_{max} - x_{min})$

2. **Handling Categorical Variables**:
   - One-hot encoding for nominal variables
   - Label encoding for ordinal variables

3. **Missing Values**: Remove or impute before clustering

### Convergence Criteria

- **Tolerance**: Stop when centroid movement is less than specified threshold
- **Maximum Iterations**: Prevent infinite loops
- **Inertia Stability**: Stop when WCSS improvement is minimal

---

## Advantages and Limitations

### Advantages

1. **Simple and Intuitive**: Easy to understand and implement
2. **Scalable**: Efficient for large datasets
3. **Guaranteed Convergence**: Always converges to a local minimum
4. **Fast**: Linear time complexity O(nkd) per iteration
5. **Memory Efficient**: Only stores centroids and cluster assignments

### Limitations

1. **Local Optima**: May converge to suboptimal solutions
2. **Sensitive to Initialization**: Different starting points can lead to different results
3. **Assumes Spherical Clusters**: Works best with isotropic, spherical clusters
4. **Fixed Number of Clusters**: Requires specifying k beforehand
5. **Sensitive to Outliers**: Outliers can significantly affect centroid positions
6. **Curse of Dimensionality**: Performance degrades in high-dimensional spaces

---

## Advanced Variants

### K-Means++

Improved initialization strategy that spreads initial centroids apart:

1. Choose first centroid randomly
2. For each subsequent centroid, choose with probability proportional to distance from nearest existing centroid
3. Reduces likelihood of poor initialization

### Mini-Batch K-Means

Memory-efficient variant that uses random subsets of data:

- Processes data in small batches
- Useful for very large datasets
- Slightly lower quality but much faster

### Fuzzy C-Means

Soft clustering where points can belong to multiple clusters with different degrees of membership.

### Kernel K-Means

Applies K-Means in a higher-dimensional feature space using kernel functions.

---

## Implementation Example

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Finding optimal k using elbow method
wcss = []
K_range = range(1, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot elbow curve and select optimal k
# optimal_k = elbow_point(wcss)

# Final clustering
kmeans = KMeans(
    n_clusters=optimal_k,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)

# Fit and predict
cluster_labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# Evaluation metrics
silhouette_avg = silhouette_score(X_scaled, cluster_labels)
calinski_score = calinski_harabasz_score(X_scaled, cluster_labels)
```

---
