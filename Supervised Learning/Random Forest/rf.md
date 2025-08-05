# Random Forests

## Introduction

Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training.

![Random Forest AUC](/images/rf.png)

## Fundamental Concepts

### Ensemble Learning

Random Forests use two key ensemble techniques:

1. **Bagging (Bootstrap Aggregating)**: Creates multiple training sets by sampling with replacement
2. **Feature Randomization**: Selects random subset of features at each split

### Key Components

- **Multiple Decision Trees**: Each tree is built independently
- **Random Subspace Method**: Different feature subsets for each tree
- **Voting/Averaging**: Combines predictions from all trees

## Mathematical Foundation

### Bootstrap Sampling

For each tree t in the forest:
$$D_t = Sample_{with\_replacement}(D, n)$$
Where:

- D is the original dataset
- n is the size of the original dataset
- D_t is the bootstrap sample for tree t

### Feature Selection

At each node, select m features randomly:
$$m = \begin{cases}
\sqrt{p} & \text{for classification} \\
p/3 & \text{for regression}
\end{cases}$$
Where p is the total number of features

### Aggregation

#### For Classification
Mode of the predictions:
$$\hat{y} = mode(h_1(x), h_2(x), ..., h_T(x))$$

#### For Regression
Mean of the predictions:
$$\hat{y} = \frac{1}{T}\sum_{t=1}^{T} h_t(x)$$
Where:
- T is the number of trees
- h_t(x) is the prediction of the t-th tree

### Out-of-Bag Error
Estimated using samples not included in bootstrap:
$$OOB_{error} = \frac{1}{n}\sum_{i=1}^{n} I(\hat{y}_i^{oob} \neq y_i)$$

### Feature Importance
Calculated using mean decrease in impurity:
$$Importance(f_j) = \frac{1}{T}\sum_{t=1}^{T}\sum_{n \in N_t} w_n \Delta i(n, f_j)$$
Where:
- N_t is the set of non-leaf nodes in tree t
- w_n is the weighted number of samples reaching node n
- Δi is the decrease in impurity

## Algorithm Steps

### Training Phase
1. For each tree in the forest:
   - Create bootstrap sample from training data
   - While nodes can be split:
     - Select m random features
     - Find best split among these features
     - Split node
   - Grow tree to maximum size (no pruning)

### Prediction Phase
1. For a new sample x:
   - Get prediction from each tree
   - Aggregate predictions:
     - Classification: Take majority vote
     - Regression: Calculate mean

## Implementation Considerations

### Hyperparameters

#### Forest-Level Parameters
- `n_estimators`: Number of trees in the forest
- `bootstrap`: Whether to use bootstrap sampling
- `oob_score`: Whether to use out-of-bag samples

#### Tree-Level Parameters
- `max_features`: Size of random feature subset
- `max_depth`: Maximum depth of trees
- `min_samples_split`: Minimum samples for splitting
- `min_samples_leaf`: Minimum samples at leaf

### Parallel Processing
Random Forests can be parallelized at two levels:
1. Tree level: Build trees in parallel
2. Node level: Find best splits in parallel

## Advantages and Limitations

### Advantages
1. Reduces overfitting through averaging
2. Handles high-dimensional data well
3. Provides feature importance rankings
4. Built-in validation through OOB error
5. Handles missing values effectively
6. No feature scaling required

### Limitations
1. Less interpretable than single decision trees
2. Computationally intensive
3. May overfit on noisy classification tasks
4. Not optimal for linear relationships
5. Requires more memory than single trees

## Advanced Concepts

### Variable Importance Measures

#### Mean Decrease Impurity (MDI)
Based on total decrease in node impurity averaged over all trees:
```python
importance = model.feature_importances_
```

#### Mean Decrease Accuracy (MDA)
Based on decrease in accuracy when a feature is permuted:
```python
from sklearn.inspection import permutation_importance
```

## Implementation Example

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification Forest
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True
)

# Regression Forest
reg = RandomForestRegressor(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='auto',
    bootstrap=True,
    oob_score=True
)

# Training
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Feature Importance
importances = model.feature_importances_
```
