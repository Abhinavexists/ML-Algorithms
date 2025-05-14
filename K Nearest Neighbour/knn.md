# K-Nearest Neighbors (KNN) Algorithm

## Introduction
The K-Nearest Neighbors (KNN) algorithm is one of the simplest yet powerful machine learning algorithms used for both classification and regression tasks. It belongs to the family of instance-based, non-parametric learning methods and is based on the principle that similar objects exist in close proximity.

![KNN Visualization](/images/knn.png)

## Fundamental Concept
The core idea behind KNN is remarkably intuitive: objects that are "similar" tend to belong to the same category. In the context of the algorithm:
- For classification: An object is classified by a plurality vote of its neighbors
- For regression: The object's value is the average of its k nearest neighbors' values

## Mathematical Foundation

### Distance Metrics
The notion of "nearness" is quantified using distance metrics. The most common metrics include:

#### Euclidean Distance
The straight-line distance between two points in Euclidean space:

$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

#### Manhattan Distance
The sum of absolute differences between coordinates:

$$d(x, y) = \sum_{i=1}^{n} |x_i - y_i|$$

#### Minkowski Distance
A generalization of both Euclidean and Manhattan:

$$d(x, y) = \left(\sum_{i=1}^{n} |x_i - y_i|^p\right)^{1/p}$$

Where:
- p = 1: Manhattan distance
- p = 2: Euclidean distance

### Decision Function

#### For Classification
The class label $\hat{y}$ for a query point $x$ is determined by:

$$\hat{y} = \arg\max_{c \in \{1,\ldots,C\}} \sum_{i \in N_k(x)} I(y_i = c)$$

Where:
- $N_k(x)$ represents the set of k nearest neighbors of query point $x$
- $I(y_i = c)$ is 1 if $y_i = c$ and 0 otherwise

#### For Regression
The predicted value $\hat{y}$ for a query point $x$ is:

##### Uniform Weights
$$\hat{y} = \frac{1}{k} \sum_{i \in N_k(x)} y_i$$

##### Distance-Weighted
$$\hat{y} = \frac{\sum_{i \in N_k(x)} w_i y_i}{\sum_{i \in N_k(x)} w_i}$$

Where $w_i = \frac{1}{d(x, x_i)^2}$, giving higher weights to closer neighbors.

## Algorithm Steps

### Training Phase
1. Store all training samples $(x_i, y_i)$ where $x_i$ is a feature vector and $y_i$ is the corresponding label/value

### Prediction Phase
1. For a query point $x$:
   - Compute distance from $x$ to all training samples
   - Select the k closest training samples (k-nearest neighbors)
   - For classification: Assign the majority class of the k neighbors
   - For regression: Calculate the mean/weighted mean of the k neighbors' target values

## Implementation Considerations

### Choosing k
- Small k: High variance, low bias (more sensitive to noise)
- Large k: Low variance, high bias (smoother decision boundaries)
- Typically odd values are used for classification to avoid ties
- k is often determined through cross-validation

### Feature Scaling
Since KNN uses distance metrics, it's essential to normalize/standardize features to ensure that no single feature dominates distance calculations:

#### Min-Max Scaling
$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

#### Z-Score Standardization
$$x_{scaled} = \frac{x - \mu}{\sigma}$$

### Computational Efficiency
Naive KNN requires computing distances to all training samples for each prediction, which is computationally expensive for large datasets. Optimization strategies include:

- **KD-Trees**: Space-partitioning data structures for efficient nearest neighbor search in low-dimensional spaces
- **Ball Trees**: Hierarchical data structures effective for high-dimensional data
- **Locality-Sensitive Hashing (LSH)**: Approximates nearest neighbors search in high-dimensional spaces

## Theoretical Properties

### Convergence
As the number of training samples approaches infinity and k is allowed to increase in a controlled manner (k → ∞, k/n → 0), KNN converges to the optimal Bayes classifier.

### Curse of Dimensionality
In high dimensions:
- The concept of "nearest" becomes less meaningful
- Data points tend to be equidistant from each other
- More training data is required to maintain prediction accuracy

## Advantages and Limitations

### Advantages
- No assumptions about the underlying data distribution
- Simple to understand and implement
- Naturally handles multi-class classification
- Can learn complex decision boundaries
- No explicit training phase
- Can be adapted for both classification and regression

### Limitations
- Computationally expensive for large datasets
- Sensitive to irrelevant features and the curse of dimensionality
- Requires feature scaling
- Memory-intensive (stores all training examples)
- Optimal k selection can be challenging
- Performance degrades with imbalanced datasets

## Advanced Variants

### Weighted KNN
Neighbors are weighted by the inverse of their distance, giving more influence to closer points.

### Local Weighted Regression
A non-linear regression method that builds on KNN by fitting a linear model to the neighborhood of each query point.

### Condensed Nearest Neighbors (CNN)
Reduces the size of the training set by removing samples that don't affect the decision boundary.
