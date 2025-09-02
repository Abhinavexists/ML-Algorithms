# DBSCAN Clustering

## Introduction

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points that are closely packed while marking points that lie alone in low-density regions as outliers. Unlike K-Means and Hierarchical Clustering, DBSCAN doesn't require specifying the number of clusters beforehand and excels at handling noise and non-linear cluster shapes.

![DBSCAN Clustering Visualization](/images/dbscan.png)

---

## Fundamental Concepts

### Basic Idea

DBSCAN works by identifying regions of high density and expanding clusters from core points. It can discover clusters of arbitrary shapes and is particularly effective at identifying outliers in the data.

### Key Components

- **Core Points**: Points with sufficient neighbors within a specified radius
- **Border Points**: Points that are reachable from core points but don't have enough neighbors
- **Outliers (Noise)**: Points that are neither core nor border points
- **Epsilon (ε)**: The radius around each point to search for neighbors
- **MinPts**: Minimum number of points required to form a core point

---

## Mathematical Foundation

### Density-Based Clustering

DBSCAN defines clusters based on the concept of density-reachability:

#### Epsilon-Neighborhood

For a point $p$, the ε-neighborhood is defined as:
$$N_\epsilon(p) = \{q \in D : d(p, q) \leq \epsilon\}$$

Where $D$ is the dataset and $d(p, q)$ is the distance between points $p$ and $q$.

#### Core Point

A point $p$ is a core point if:
$$|N_\epsilon(p)| \geq MinPts$$

#### Border Point

A point $q$ is a border point if:
$$|N_\epsilon(q)| < MinPts$$

But $q$ is reachable from some core point $p$ (i.e., $q \in N_\epsilon(p)$).

#### Outlier (Noise)

A point $r$ is an outlier if it's neither a core point nor a border point.

### Distance Metrics

#### Euclidean Distance

$$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$

#### Manhattan Distance

$$d(p, q) = \sum_{i=1}^{n} |p_i - q_i|$$

#### Cosine Distance

$$d(p, q) = 1 - \frac{\sum_{i=1}^{n} p_i q_i}{\sqrt{\sum_{i=1}^{n} p_i^2} \sqrt{\sum_{i=1}^{n} q_i^2}}$$

---

## Algorithm Steps

### Step 1: Parameter Setting

1. **Set ε (epsilon)**: The radius around each point
2. **Set MinPts**: Minimum number of points required to form a core point

### Step 2: Point Classification

For each point $p$ in the dataset:

1. **Calculate ε-neighborhood**: Find all points within radius ε of $p$
2. **Classify point type**:
   - If $|N_\epsilon(p)| \geq MinPts$: Mark as **core point**
   - If $|N_\epsilon(p)| < MinPts$ but reachable from core point: Mark as **border point**
   - Otherwise: Mark as **outlier**

### Step 3: Cluster Formation

1. **Start with unvisited points**
2. **For each unvisited core point**:
   - Create a new cluster
   - Add all density-reachable points to the cluster
   - Mark all points in the cluster as visited
3. **Continue until all core points are processed**

### Step 4: Final Assignment

- **Core points**: Assigned to their respective clusters
- **Border points**: Assigned to the cluster of the nearest core point
- **Outliers**: Remain unassigned (labeled as -1)

---

## Parameters

### Epsilon (ε)

**Definition**: The radius around each point to search for neighbors

**Selection Guidelines**:

- **Too small**: Many points become outliers, clusters become fragmented
- **Too large**: Different clusters may merge, losing cluster separation
- **Optimal value**: Often found using k-distance graph analysis

**K-Distance Graph Method**:

1. For each point, find the distance to its kth nearest neighbor
2. Sort these distances in descending order
3. Look for the "elbow" in the plot
4. The elbow point suggests a good ε value

### MinPts

**Definition**: Minimum number of points required to form a core point

**Selection Guidelines**:

- **Minimum value**: 2 (for 2D data)
- **Recommended**: 2 × number of dimensions
- **Too small**: Sensitive to noise
- **Too large**: May miss small clusters

**Common Values**:

- **2D data**: MinPts = 4-5
- **3D data**: MinPts = 6-8
- **High-dimensional data**: MinPts = 2 × dimensions

---

## Implementation Considerations

### Hyperparameters

- **eps**: The maximum distance between two samples for one to be considered as in the neighborhood of the other
- **min_samples**: The number of samples in a neighborhood for a point to be considered as a core point
- **metric**: The metric to use when calculating distance between instances
- **algorithm**: The algorithm to use for nearest neighbors search

### Data Preprocessing

1. **Feature Scaling**: Essential since DBSCAN is distance-based
   - StandardScaler: $(x - \mu) / \sigma$
   - MinMaxScaler: $(x - x_{min}) / (x_{max} - x_{min})$

2. **Handling Categorical Variables**:
   - One-hot encoding for nominal variables
   - Label encoding for ordinal variables

3. **Missing Values**: Remove or impute before clustering

### Computational Complexity

- **Time Complexity**: O(n²) in worst case, O(n log n) with spatial indexing
- **Space Complexity**: O(n) for storing cluster assignments
- **Scalability**: Can handle datasets with n < 100,000 points efficiently

---

## Advantages and Limitations

### Advantages

1. **No Need to Specify k**: Automatically determines number of clusters
2. **Handles Noise**: Identifies and separates outliers from clusters
3. **Non-Linear Clusters**: Can discover clusters of arbitrary shapes
4. **Density-Based**: Works well with clusters of varying densities
5. **Robust**: Less sensitive to initialization than K-Means
6. **Interpretable**: Clear distinction between core, border, and noise points

### Limitations

1. **Parameter Sensitivity**: Results heavily depend on ε and MinPts
2. **Density Variations**: Struggles with clusters of very different densities
3. **High-Dimensional Data**: Curse of dimensionality affects distance calculations
4. **Computational Cost**: O(n²) complexity for large datasets
5. **Border Point Assignment**: Border points may be assigned to different clusters based on order

---

## Comparison with Other Algorithms

### vs K-Means

| Aspect | K-Means | DBSCAN |
|--------|---------|---------|
| **Cluster Shape** | Spherical | Arbitrary |
| **Noise Handling** | Poor | Excellent |
| **Parameter Tuning** | Number of clusters | ε and MinPts |
| **Scalability** | O(nkd) per iteration | O(n²) |
| **Initialization** | Sensitive to centroids | Deterministic |

### vs Hierarchical Clustering

| Aspect | Hierarchical | DBSCAN |
|--------|--------------|---------|
| **Cluster Count** | Automatically determined | Automatically determined |
| **Noise Handling** | Poor | Excellent |
| **Shape Flexibility** | Good | Excellent |
| **Computational Cost** | O(n²) | O(n²) |
| **Parameter Tuning** | Linkage method | ε and MinPts |

---

## Implementation Example

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# Data preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Method 1: K-Distance Graph for ε selection
def find_optimal_eps(X, k=4):
    """Find optimal epsilon using k-distance graph"""
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors_fit = neighbors.fit(X)
    distances, indices = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, k-1])
    
    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(distances)), distances)
    plt.xlabel('Points')
    plt.ylabel(f'{k}-Distance')
    plt.title('K-Distance Graph for Epsilon Selection')
    plt.show()
    
    # Find elbow point (you can implement automatic detection)
    return distances[len(distances)//2]  # Example: use median

# Find optimal epsilon
optimal_eps = find_optimal_eps(X_scaled, k=4)

# DBSCAN clustering
dbscan = DBSCAN(
    eps=optimal_eps,
    min_samples=4,
    metric='euclidean'
)

# Fit and predict
cluster_labels = dbscan.fit_predict(X_scaled)

# Analyze results
n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
n_noise = list(cluster_labels).count(-1)

print(f'Estimated number of clusters: {n_clusters}')
print(f'Estimated number of noise points: {n_noise}')

# Visualize clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_scaled[:, 0], X_scaled[:, 1], 
                     c=cluster_labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('DBSCAN Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
```
