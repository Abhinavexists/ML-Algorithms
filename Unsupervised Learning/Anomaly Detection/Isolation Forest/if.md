# Isolation Forest

## Introduction

Isolation Forest is a tree-based anomaly detection algorithm that isolates anomalies by randomly partitioning the data space. Unlike traditional methods that construct profiles of normal data points, Isolation Forest explicitly isolates anomalies using binary trees. It is particularly effective for high-dimensional datasets and has linear time complexity, making it suitable for large-scale applications.

## Core Concepts

### Basic Idea

Isolation Forest works on the principle that anomalies are "few and different," making them easier to isolate than normal points. By randomly partitioning the data space using binary trees, anomalies require fewer splits to be isolated compared to normal points, which tend to cluster together.

### Key Components

- **Isolation Trees (iTrees)**: Binary trees that recursively partition the data
- **Path Length**: Number of splits needed to isolate a point
- **Anomaly Score**: Based on the average path length across multiple trees
- **Ensemble Method**: Uses multiple trees for robust anomaly detection

---

## Mathematical Foundation

### Isolation Tree Construction

An isolation tree is built by recursively partitioning the data:

1. **Select a feature**: Randomly choose a feature to split on
2. **Select split value**: Choose a random split value between min and max of the feature
3. **Partition data**: Divide data into left and right subsets
4. **Repeat**: Continue until stopping criteria are met

### Path Length

The path length $h(x)$ for a point $x$ in an isolation tree is the number of edges from root to the leaf containing $x$.

For a dataset of size $n$, the average path length $c(n)$ of an unsuccessful search in a binary search tree is:

$$c(n) = 2H(n-1) - 2(n-1)/n$$

Where $H(i)$ is the harmonic number: $H(i) = \sum_{k=1}^{i} \frac{1}{k}$

### Anomaly Score

The anomaly score $s(x, n)$ for a point $x$ is calculated as:

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

Where:

- $E[h(x)]$ is the average path length over all isolation trees
- $c(n)$ is the average path length of unsuccessful searches

**Score Interpretation:**

- **s(x, n) ≈ 1**: Anomalies (short path lengths)
- **s(x, n) < 0.5**: Normal points
- **s(x, n) ≈ 0.5**: Points with random path lengths

---

## Algorithm Steps

### Step 1: Parameter Selection

1. **Choose number of trees (n_estimators)**: Typically 100-200
2. **Choose subsample size**: Usually 256 (2^8)
3. **Choose contamination**: Expected proportion of anomalies

### Step 2: Build Isolation Forest

For each tree in the ensemble:

1. **Sample data**: Randomly sample subsample_size points
2. **Build isolation tree**: Recursively partition the data
3. **Store tree**: Keep the tree for scoring

### Step 3: Score Calculation

For each test point:

1. **Compute path lengths**: Get path length in each tree
2. **Average path lengths**: Calculate mean across all trees
3. **Compute anomaly score**: Use the scoring formula

### Step 4: Anomaly Classification

1. **Set threshold**: Based on contamination parameter
2. **Classify points**: Above threshold = anomaly, below = normal

---

## Parameters

### n_estimators

**Definition**: Number of isolation trees in the forest

**Guidelines**:

- **Default**: 100
- **Range**: 50-200 for most applications
- **Trade-off**: More trees = better accuracy but slower computation
- **Recommendation**: Start with 100, increase for better accuracy

### max_samples

**Definition**: Number of samples to draw for each tree

**Guidelines**:

- **Default**: 'auto' (256 for most cases)
- **Range**: 100-1000 depending on dataset size
- **Trade-off**: Larger samples = more accurate but slower
- **Recommendation**: Use default unless dataset is very large

### contamination

**Definition**: Expected proportion of anomalies in the dataset

**Guidelines**:

- **Known proportion**: Use domain knowledge
- **Unknown proportion**: Use 'auto' or statistical methods
- **Range**: 0.0 to 0.5 (0-50% anomalies)
- **Default**: 'auto'

### max_features

**Definition**: Number of features to consider for each split

**Guidelines**:

- **Default**: 1.0 (all features)
- **Range**: 0.5 to 1.0
- **Use case**: Reduce for very high-dimensional data
- **Recommendation**: Use default for most cases

---

## Implementation Considerations

### Hyperparameters

- **n_estimators**: Number of base estimators
- **max_samples**: Number of samples for each tree
- **contamination**: Proportion of anomalies
- **max_features**: Number of features for each split
- **bootstrap**: Whether to use bootstrap sampling
- **random_state**: Random seed for reproducibility

### Data Preprocessing

1. **Feature Scaling**: Not required (tree-based method)
2. **Handling Categorical Variables**:
   - One-hot encoding for nominal variables
   - Label encoding for ordinal variables
3. **Missing Values**: Handle before training

### Computational Complexity

- **Time Complexity**: O(n log n) for training, O(n) for scoring
- **Space Complexity**: O(n_estimators × max_samples)
- **Scalability**: Excellent for large datasets

---

## Minimal End-to-End Example

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import numpy as np

# Generate sample data
X, _ = make_blobs(n_samples=300, centers=2, cluster_std=[1.0, 2.0], random_state=42)

# Add outliers
np.random.seed(42)
outliers = np.random.uniform(-10, 10, (15, 2))
X = np.vstack([X, outliers])

# Preprocess data (optional for Isolation Forest)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply Isolation Forest
isolation_forest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
labels = isolation_forest.fit_predict(X_scaled)
scores = isolation_forest.decision_function(X_scaled)

# Results
n_anomalies = np.sum(labels == -1)
print(f"Detected {n_anomalies} anomalies out of {len(X)} points")
print(".1f")
print(f"Average anomaly score: {np.mean(scores):.3f}")
```
