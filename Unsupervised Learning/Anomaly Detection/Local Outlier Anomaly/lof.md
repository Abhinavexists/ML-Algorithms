# Local Outlier Factor (LOF)

## Introduction

Local Outlier Factor (LOF) is a density-based anomaly detection algorithm that identifies outliers by comparing the local density of a point with the local densities of its neighbors. Unlike global anomaly detection methods, LOF can detect anomalies that are outliers relative to their local neighborhood, making it particularly effective for datasets with varying densities and detecting local anomalies.

## Core Concepts

### Basic Idea

LOF measures the degree to which a data point is an outlier by comparing its local density with the local densities of its k-nearest neighbors. Points with significantly lower density than their neighbors are considered anomalies.

### Key Components

- **k-Nearest Neighbors (kNN)**: The k closest points to a given point
- **Local Reachability Density (LRD)**: Local density of a point
- **Local Outlier Factor (LOF)**: Ratio of local density of point to densities of its neighbors
- **Reachability Distance**: Modified distance measure that prevents "masking" effects

---

## Mathematical Foundation

### k-Distance

For a point $p$ and a positive integer $k$, the k-distance of $p$ is defined as:

$$d_k(p) = \max_{q \in N_k(p)} d(p, q)$$

Where $N_k(p)$ is the set of k-nearest neighbors of $p$.

### k-Distance Neighborhood

The k-distance neighborhood of $p$ is:

$$N_{k-distance}(p) = \{q \in D : d(p, q) \leq d_k(p)\}$$

### Reachability Distance

The reachability distance from point $o$ to point $p$ is:

$$reachability\_distance_k(o, p) = \max(d_k(o), d(o, p))$$

This prevents the "masking" effect where dense clusters can hide outliers.

### Local Reachability Density (LRD)

The local reachability density of point $p$ is:

$$lrd_k(p) = \frac{|N_k(p)|}{\sum_{o \in N_k(p)} reachability\_distance_k(o, p)}$$

Higher LRD values indicate higher local density.

### LOF Calculation

The local outlier factor of point $p$ is:

$$LOF_k(p) = \frac{\sum_{o \in N_k(p)} \frac{lrd_k(o)}{lrd_k(p)}}{|N_k(p)|}$$

**Interpretation:**

- **LOF = 1**: Point has similar density to its neighbors (normal)
- **LOF < 1**: Point has higher density than its neighbors (dense region)
- **LOF > 1**: Point has lower density than its neighbors (potential anomaly)

---

## Algorithm Steps

### Step 1: Parameter Selection

1. **Choose k**: Number of nearest neighbors to consider
2. **Select distance metric**: Euclidean, Manhattan, etc.

### Step 2: Compute k-Nearest Neighbors

For each point $p$ in the dataset:

1. **Find k-nearest neighbors**: $N_k(p)$
2. **Compute k-distance**: $d_k(p)$
3. **Store neighbor relationships**

### Step 3: Calculate Reachability Distances

For each point $p$ and its neighbors:

1. **Compute reachability distance** to each neighbor
2. **Store reachability distances** for LRD calculation

### Step 4: Compute Local Reachability Density (LRD)

For each point $p$:

1. **Calculate LRD** using formula above
2. **Higher LRD** = higher local density

### Step 5: Compute LOF Scores

For each point $p$:

1. **Calculate LOF** using formula above
2. **Compare LOF with threshold** to identify anomalies

### Step 6: Anomaly Classification

- **LOF > threshold**: Point is an anomaly
- **LOF ≤ threshold**: Point is normal

---

## Parameters

### k (Number of Neighbors)

**Definition**: Number of nearest neighbors to consider for density estimation

**Selection Guidelines**:

- **Too small k**: Sensitive to noise, many false positives
- **Too large k**: May miss local anomalies, less sensitive
- **Optimal k**: Depends on dataset size and local structure
- **Common range**: 5-20 for most datasets

**Selection Methods**:

1. **Rule of thumb**: k = min(20, n/10) where n is dataset size
2. **Cross-validation**: Test different k values and evaluate
3. **Domain knowledge**: Use domain expertise to guide selection

### Distance Metric

**Definition**: Metric used to measure distance between points

**Common Options**:

- **Euclidean**: Standard distance, works well for continuous data
- **Manhattan**: Good for high-dimensional data
- **Cosine**: Better for text/document data
- **Hamming**: For categorical/binary data

### Contamination Parameter

**Definition**: Expected proportion of anomalies in the dataset

**Guidelines**:

- **Known contamination**: Use domain knowledge
- **Unknown contamination**: Use statistical methods
- **Default**: 0.1 (10% expected anomalies)

---

## Minimal End-to-End Example

```python
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import numpy as np

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=2, cluster_std=[1.0, 2.0], random_state=42)

# Add outliers
np.random.seed(42)
outliers = np.random.uniform(-10, 10, (15, 2))
X = np.vstack([X, outliers])

# Preprocess data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
labels = lof.fit_predict(X_scaled)
scores = lof.negative_outlier_factor_

# Results
n_anomalies = np.sum(labels == -1)
print(f"Detected {n_anomalies} anomalies out of {len(X)} points")
print(f"Anomaly percentage: {100 * n_anomalies / len(X):.1f}%")
print(f"Average LOF score: {np.mean(scores):.3f}")
```
