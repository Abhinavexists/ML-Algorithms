# Gradient Boosting

## Introduction

Gradient Boosting is a powerful ensemble learning technique that builds models sequentially, where each new model corrects the errors made by the previous models. It combines weak learners (typically decision trees) to create a strong predictive model.

![Gradient Boosting Visualization](/images/gb.png)

## Fundamental Concepts

### Sequential Learning

Unlike bagging methods that build models independently, Gradient Boosting builds models sequentially:

1. **Sequential Training**: Each new model is trained to correct the residual errors of the previous ensemble
2. **Gradient Descent**: Uses gradient descent to minimize the loss function by adding new models
3. **Weak Learners**: Typically uses shallow decision trees (stumps) as base learners

### Key Components

- **Base Learners**: Usually decision trees with limited depth (1-8 levels)
- **Loss Function**: Guides the learning process (e.g., squared error for regression, log-loss for classification)
- **Learning Rate**: Controls the contribution of each tree to the final ensemble
- **Regularization**: Prevents overfitting through various techniques

## Mathematical Foundation

### Algorithm Overview

For a regression problem, Gradient Boosting works as follows:

1. **Initialize** with a constant prediction:
   $$F_0(x) = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, \gamma)$$

2. **For each iteration** m = 1 to M:

   a) **Compute pseudo-residuals**:
   $$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$$

   b) **Train weak learner** $h_m(x)$ on residuals $r_{im}$

   c) **Find optimal step size**:
   $$\gamma_m = \arg\min_\gamma \sum_{i=1}^{n} L(y_i, F_{m-1}(x_i) + \gamma h_m(x_i))$$

   d) **Update the model**:
   $$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$

3. **Final prediction**:
   $$F(x) = F_0(x) + \sum_{m=1}^{M} \gamma_m h_m(x)$$

### Loss Functions

#### For Regression

- **Squared Error**: $L(y, F(x)) = \frac{1}{2}(y - F(x))^2$
- **Absolute Error**: $L(y, F(x)) = |y - F(x)|$
- **Huber Loss**: Combines squared and absolute error for robustness

#### For Classification

- **Log-loss**: $L(y, F(x)) = \log(1 + e^{-yF(x)})$
- **Exponential Loss**: $L(y, F(x)) = e^{-yF(x)}$ (similar to AdaBoost)

### Learning Rate and Regularization

The learning rate $\nu$ controls the contribution of each tree:
$$F_m(x) = F_{m-1}(x) + \nu \gamma_m h_m(x)$$

Where $0 < \nu \leq 1$. Smaller learning rates require more trees but often lead to better generalization.

## Algorithm Steps

### Training Phase

1. **Initialize** the model with a constant value
2. **For each boosting iteration**:
   - Calculate pseudo-residuals (gradients of loss function)
   - Train a weak learner on the residuals
   - Find the optimal coefficient for the new learner
   - Update the ensemble by adding the weighted new learner
3. **Repeat** until convergence or maximum iterations reached

### Prediction Phase

1. **For a new sample x**:
   - Start with the initial constant prediction
   - Add the weighted contributions of all trees in sequence
   - Return the final aggregated prediction

## Implementation Considerations

### Hyperparameters

#### Tree-Specific Parameters

- `max_depth`: Depth of individual trees (typically 3-8)
- `min_samples_split`: Minimum samples required to split
- `min_samples_leaf`: Minimum samples at leaf nodes

#### Boosting Parameters

- `n_estimators`: Number of boosting stages (trees)
- `learning_rate`: Shrinks contribution of each tree
- `subsample`: Fraction of samples used for each tree
- `max_features`: Number of features considered at each split

#### Loss Function Parameters

- `loss`: Loss function to optimize
- `criterion`: Split quality measure

### Preventing Overfitting

1. **Learning Rate**: Lower values require more trees but reduce overfitting
2. **Subsampling**: Use random subsets of data for each tree
3. **Feature Sampling**: Use random subsets of features
4. **Early Stopping**: Monitor validation error to stop training
5. **Tree Constraints**: Limit tree depth and minimum samples

## Advantages and Limitations

### Advantages

1. **High Predictive Accuracy**: Often achieves excellent performance
2. **Feature Importance**: Provides natural feature importance rankings
3. **Handles Mixed Data Types**: Works with both numerical and categorical features
4. **Robust to Outliers**: Especially with robust loss functions
5. **No Feature Scaling Required**: Tree-based method doesn't require normalization
6. **Handles Missing Values**: Can work with missing data

### Limitations

1. **Sensitive to Overfitting**: Requires careful hyperparameter tuning
2. **Sequential Training**: Cannot be easily parallelized
3. **Computationally Intensive**: Training can be slow for large datasets
4. **Hyperparameter Sensitivity**: Performance highly dependent on parameter choices
5. **Memory Requirements**: Stores all trees in memory
6. **Difficult Interpretation**: Less interpretable than single decision trees

## Implementation Analysis

### Dataset Applications

Our implementation demonstrates Gradient Boosting on two different problems:

#### 1. Holiday Package Prediction (Classification)

- **Dataset**: Travel.csv (4,888 samples, 20 features)
- **Problem**: Predict whether customers will purchase wellness tourism packages
- **Features**: Demographics, behavioral data, and travel preferences
- **Target**: Binary classification (package purchased: Yes/No)

#### 2. Used Car Price Prediction (Regression)

- **Dataset**: cardekho_imputated.csv (15,411 samples, 13 features)
- **Problem**: Predict selling price of used cars
- **Features**: Car specifications, age, mileage, seller type, etc.
- **Target**: Continuous price prediction

### Performance Results

#### Classification Results (Holiday Package Prediction)

**Initial Model Performance**:

- **Training Accuracy**: 0.8936
- **Test Accuracy**: 0.8589
- **Training F1-score**: 0.8809
- **Test F1-score**: 0.8369
- **Training Precision**: 0.8883
- **Test Precision**: 0.8046
- **Training Recall**: 0.4911
- **Test Recall**: 0.3665
- **Training ROC-AUC**: 0.7385
- **Test ROC-AUC**: 0.6724

**Optimized Model Performance** (after hyperparameter tuning):

- **Training Accuracy**: 1.0000
- **Test Accuracy**: 0.9315
- **Training F1-score**: 1.0000
- **Test F1-score**: 0.9265
- **Training Precision**: 1.0000
- **Test Precision**: 0.9697
- **Training Recall**: 1.0000
- **Test Recall**: 0.6702
- **Training ROC-AUC**: 1.0000
- **Test ROC-AUC**: 0.8325

**Best Parameters**:

```python
{
    'subsample': 1.0,
    'n_estimators': 500,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_depth': 10,
    'loss': 'log_loss',
    'learning_rate': 0.05,
    'criterion': 'squared_error'
}
```

#### Regression Results (Used Car Price Prediction)

**Initial Model Performance**:

- **Training RMSE**: 204,944.5104
- **Test RMSE**: 254,830.9686
- **Training MAE**: 111,709.5558
- **Test MAE**: 126,065.3619
- **Training R²**: 0.9482
- **Test R²**: 0.9137

**Optimized Model Performance** (after hyperparameter tuning):

- **Training RMSE**: 61,599.5516
- **Test RMSE**: 301,359.1086
- **Training MAE**: 37,085.1104
- **Test MAE**: 101,678.3296
- **Training R²**: 0.9953
- **Test R²**: 0.8794

**Best Parameters**:

```python
{
    'n_estimators': 500,
    'min_samples_split': 8,
    'max_depth': 8,
    'loss': 'huber',
    'criterion': 'friedman_mse'
}
```

### Model Comparison

#### Classification Comparison

In the holiday package prediction task, Gradient Boosting performed competitively:

- **Random Forest**: 0.9059 accuracy, 0.7730 ROC-AUC
- **Gradient Boosting**: 0.8589 accuracy, 0.6724 ROC-AUC (initial)
- **Gradient Boosting** (tuned): 0.9315 accuracy, 0.8325 ROC-AUC
- **Decision Tree**: 0.8855 accuracy, 0.8079 ROC-AUC
- **AdaBoost**: 0.8405 accuracy, 0.6114 ROC-AUC

#### Regression Comparison

In the used car price prediction task:

- **Random Forest**: 226,217 RMSE, 0.9320 R²
- **Gradient Boosting**: 254,831 RMSE, 0.9137 R² (initial)
- **K-Neighbors**: 253,072 RMSE, 0.9149 R²
- **Decision Tree**: 301,565 RMSE, 0.8792 R²
- **AdaBoost**: 433,294 RMSE, 0.7506 R²

### Key Implementation Insights

1. **Hyperparameter Tuning Critical**: The performance difference between default and tuned parameters was substantial, especially for classification (ROC-AUC improved from 0.6724 to 0.8325).

2. **Learning Rate Trade-off**: The optimal learning rate of 0.05 for classification provided a good balance between training speed and model performance.

3. **Tree Depth Matters**: Moderate tree depths (8-10) performed best, avoiding both underfitting and overfitting.

4. **Feature Sampling**: Using 'sqrt' for max_features helped prevent overfitting while maintaining good performance.

5. **Loss Function Selection**:
   - For classification: 'log_loss' provided better probability estimates
   - For regression: 'huber' loss showed robustness to outliers

6. **Overfitting Tendency**: The perfect training scores (1.0) in the optimized classification model suggest potential overfitting, despite good test performance.

## Relationship to Other Algorithms

### Comparison with Other Ensemble Methods

- **vs Random Forest**: Sequential vs parallel training, bias-variance trade-off
- **vs AdaBoost**: More flexible loss functions, better handling of outliers
- **vs XGBoost**: Similar concept but XGBoost includes additional optimizations

### Theoretical Connections

- **Gradient Descent**: Applies gradient descent in function space
- **Forward Stagewise Additive Modeling**: Builds models incrementally
- **Functional Gradient Descent**: Extends gradient descent to function optimization

## Best Practices

### Hyperparameter Tuning Strategy

1. **Start with defaults** and establish baseline performance
2. **Tune learning rate** and n_estimators together
3. **Optimize tree structure** (max_depth, min_samples_split)
4. **Add regularization** (subsample, max_features)
5. **Use cross-validation** for robust parameter selection

### Prevent Overfitting

1. **Use validation curves** to monitor training vs validation performance
2. **Implement early stopping** based on validation metrics
3. **Apply subsampling** to introduce randomness
4. **Limit tree complexity** with depth and sample constraints

### Production Considerations

1. **Model Serialization**: Save trained models for deployment
2. **Feature Consistency**: Ensure same preprocessing pipeline
3. **Monitoring**: Track model performance over time
4. **Retraining Strategy**: Plan for model updates with new data
