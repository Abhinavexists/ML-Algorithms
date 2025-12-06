# Logistic Regression (LoR)

## Introduction

Logistic Regression models the probability of a binary (or multiclass) outcome using a linear decision function passed through a sigmoid (binary) or softmax (multiclass) link. Despite its name, it is a classification algorithm. It is widely used for its interpretability, efficiency, and well-calibrated probabilities.

---

## Core Concepts

- **Decision function**: $z = w_0 + \sum_{j=1}^{p} w_j x_j$
- **Sigmoid (binary)**: $\sigma(z) = \frac{1}{1 + e^{-z}}$, models $P(y=1\mid x) = \sigma(z)$
- **Softmax (multiclass)**: $P(y=k\mid x) = \frac{e^{z_k}}{\sum_{c} e^{z_c}}$
- **Loss**: Negative log-likelihood (log-loss / cross-entropy)
- **Regularization**: L2 (Ridge), L1 (Lasso), Elastic-Net to prevent overfitting

---

## Mathematical Foundation

### Binary Logistic Regression

Probability of the positive class:

$$P(y=1\mid x) = \sigma(\mathbf{w}^T\mathbf{x}) = \frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}$$

Log-loss for a dataset $(x_i, y_i)$:

$$\mathcal{L}(\mathbf{w}) = -\sum_{i=1}^n \big[ y_i \log \hat{p}_i + (1-y_i) \log (1-\hat{p}_i) \big] \quad \text{where } \hat{p}_i = \sigma(\mathbf{w}^T x_i)$$

Regularized objective (L2):

$$J(\mathbf{w}) = \mathcal{L}(\mathbf{w}) + \lambda \|\mathbf{w}\|_2^2$$

Gradient:

$$\nabla J(\mathbf{w}) = X^T(\hat{\mathbf{p}} - \mathbf{y}) + 2\lambda \mathbf{w}$$

### Multiclass (Softmax Regression)

For classes $k=1,\ldots,K$:

$$P(y=k\mid x) = \frac{e^{\mathbf{w}_k^T x}}{\sum_{c=1}^K e^{\mathbf{w}_c^T x}}$$

Cross-entropy loss:

$$\mathcal{L}(W) = - \sum_{i=1}^n \sum_{k=1}^K \mathbb{1}\{y_i=k\} \log P(y_i=k\mid x_i)$$

---

## Assumptions and Properties

- Linear decision boundary in feature space
- Well-calibrated probabilities with proper regularization
- Requires minimal preprocessing; scaling helps with optimization
- Sensitive to multicollinearity; L2/L1 helps

---

## Handling Class Imbalance

- **Class weights**: Use `class_weight='balanced'` or custom weights
- **Threshold tuning**: Adjust decision threshold from 0.5 using precision-recall trade-offs
- **Resampling**: Over-sampling minority (SMOTE) or under-sampling majority
- **Metrics**: Prefer ROC-AUC, PR-AUC, F1, balanced accuracy over accuracy

---

## Evaluation Metrics

- **Accuracy**, **Precision**, **Recall**, **F1-score**
- **ROC-AUC**, **PR-AUC**
- **Log-loss**
- **Calibration**: reliability curves, Brier score

---

## scikit-learn Implementations

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.datasets import make_classification
import numpy as np

# Synthetic binary dataset
X, y = make_classification(
    n_samples=3000, n_features=10, n_informative=5, n_redundant=2,
    weights=[0.7, 0.3], random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

# 1) Binary Logistic Regression with L2 (liblinear for small datasets, lbfgs/saga for larger)
binary_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=1000))
])

binary_lr.fit(X_train, y_train)
y_pred = binary_lr.predict(X_test)
y_prob = binary_lr.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

# 2) Imbalanced data handling with class_weight
imb_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000))
])
imb_lr.fit(X_train, y_train)

# 3) Multiclass (softmax)
X_mc, y_mc = make_classification(
    n_samples=2000, n_features=8, n_classes=3, n_informative=6,
    n_redundant=0, random_state=0
)
Xmc_tr, Xmc_te, ymc_tr, ymc_te = train_test_split(X_mc, y_mc, test_size=0.25, random_state=0, stratify=y_mc)

softmax_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=1000))
])
softmax_lr.fit(Xmc_tr, ymc_tr)
print(f"Multiclass accuracy: {softmax_lr.score(Xmc_te, ymc_te):.3f}")
```

---

## Practical Tips

- Scale features for faster, more stable convergence
- Try `lbfgs` for small-to-medium dense data, `saga` for large/elastic-net, `liblinear` for small/rare features
- Use `C` (inverse of regularization strength) to control complexity; smaller `C` -> stronger regularization
- Calibrate probabilities when needed (`CalibratedClassifierCV`)
- Tune threshold using ROC/PR curves for desired precision-recall tradeoff

---

## Advantages

- Interpretable coefficients (log-odds) and class probabilities
- Efficient, robust, and strong baseline for many tasks
- Supports L1/L2/Elastic-Net and multiclass (one-vs-rest or multinomial)

## Limitations

- Linear decision boundaries (needs features/expansion for complex relations)
- Performance degrades with severe multicollinearity without regularization
- Can be sensitive to overlapping classes and outliers

---

## Related Notebooks in this Folder

- `Logistic_Regression.ipynb`
- `Multiclass_Logistic_Regression.ipynb`
- `LogReg_ROC_AUC_Score.ipynb`
- `LogReg_Imb_Dataset.ipynb`
