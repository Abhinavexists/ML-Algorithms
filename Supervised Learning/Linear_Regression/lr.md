# Linear Regression (LR)

## Introduction

Linear Regression models the relationship between a scalar target variable and one or more explanatory variables by fitting a linear function. It is foundational for regression tasks and serves as a strong baseline for many real-world problems.

---

## Core Concepts

- **Simple Linear Regression**: One feature, line fit: $y = w_0 + w_1 x$
- **Multiple Linear Regression**: Multiple features: $y = w_0 + \sum_{j=1}^{p} w_j x_j$
- **Polynomial Regression**: Linear in parameters after feature expansion (e.g., degree-2: add $x^2$, interactions)
- **Regularized Regression**: Ridge (L2), Lasso (L1) to control overfitting

---

## Mathematical Foundation

### Objective (Ordinary Least Squares)

Minimize Residual Sum of Squares (RSS):

$$\min_\mathbf{w} \; J(\mathbf{w}) = \| \mathbf{y} - X\mathbf{w} \|_2^2$$

Where:

- $X \in \mathbb{R}^{n\times p}$ is the design matrix (include a column of ones for intercept)
- $\mathbf{y} \in \mathbb{R}^{n}$ is the target vector
- $\mathbf{w} \in \mathbb{R}^{p}$ are coefficients (including intercept if not separated)

### Closed-Form (Normal Equation)

If $X^T X$ is invertible:

$$\hat{\mathbf{w}} = (X^T X)^{-1} X^T \mathbf{y}$$

With L2 regularization (Ridge) and regularization strength $\lambda$:

$$\hat{\mathbf{w}} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$$

### Gradient Descent (iterative)

Update rule with learning rate $\eta$:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \; \nabla J(\mathbf{w}) \quad \text{where} \quad \nabla J(\mathbf{w}) = -2 X^T (\mathbf{y} - X\mathbf{w})$$

Variants: Batch GD, Stochastic GD (SGD), Mini-batch GD.

### Assumptions (OLS)

- Linearity: relationship is linear in parameters
- Independence: errors are independent
- Homoscedasticity: constant error variance
- Normality: errors are normally distributed (for inference)
- No/low multicollinearity among predictors

---

## Feature Engineering and Preprocessing

- **Scaling**: Not required for OLS solution; recommended for GD/regularization
- **Polynomial features**: For non-linear patterns while keeping linear-in-parameters
- **Categorical variables**: One-hot encode
- **Interaction terms**: Capture feature interactions
- **Outliers**: Can distort coefficients; consider robust variants

---

## Evaluation Metrics

- **R² (Coefficient of Determination)**
- **MAE (Mean Absolute Error)**
- **MSE (Mean Squared Error)**
- **RMSE (Root Mean Squared Error)**
- **Adjusted R²** for multiple regression

---

## scikit-learn Implementations

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Example synthetic data
rng = np.random.RandomState(42)
X = rng.rand(500, 1) * 10
y = 3.0 + 2.5 * X[:, 0] + rng.normal(0, 1.2, size=500)
X = X  # shape (n, 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# 1) Ordinary Least Squares
ols = LinearRegression()
ols.fit(X_train, y_train)
y_pred = ols.predict(X_test)
print(f"OLS R^2: {r2_score(y_test, y_pred):.3f}")
print(f"OLS RMSE: {mean_squared_error(y_test, y_pred, squared=False):.3f}")

# 2) Polynomial Regression (degree 2)
poly_model = Pipeline([
    ("poly", PolynomialFeatures(degree=2, include_bias=False)),
    ("linreg", LinearRegression())
])
poly_model.fit(X_train, y_train)
y_pred_poly = poly_model.predict(X_test)
print(f"Poly(2) R^2: {r2_score(y_test, y_pred_poly):.3f}")

# 3) Ridge (L2)
ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0, random_state=0))
])
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)
print(f"Ridge R^2: {r2_score(y_test, y_pred_ridge):.3f}")

# 4) Lasso (L1)
lasso_model = Pipeline([
    ("scaler", StandardScaler()),
    ("lasso", Lasso(alpha=0.1, random_state=0, max_iter=10000))
])
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)
print(f"Lasso R^2: {r2_score(y_test, y_pred_lasso):.3f}")

# Cross-validation example
cv_scores = cross_val_score(ols, X, y, scoring="r2", cv=5)
print(f"OLS CV R^2 (mean±std): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

---

## Practical Tips

- Add an intercept term (or let the library handle it)
- Inspect residual plots to check assumptions
- Use regularization if features are many or correlated
- Standardize features when using GD, Ridge, or Lasso
- Prefer cross-validation for model selection
- Consider log/Box-Cox transforms for skewed targets

---

## Advantages

- Simple, fast, and interpretable
- Strong baseline; easy to deploy
- Closed-form solution for OLS

## Limitations

- Sensitive to outliers and multicollinearity
- Captures only linear relationships (unless features engineered)
- Assumption violations degrade inference reliability

---

## Related Notebooks in this Folder

- `Simple_Linear_Regression.ipynb`
- `Multiple_Linear_Regession.ipynb`
- `Polynomial_Regression.ipynb`
- `Ridge_Lasso_Regression.ipynb`
- `Model Training.ipynb`
