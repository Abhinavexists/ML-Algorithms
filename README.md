# ML-Algorithms

Complete machine learning algorithms implementation repository with step-by-step examples, from fundamentals to advanced techniques.

## Supervised Learning

### 1. Linear Models (Foundation)

| Order | Algorithm | Description | Notebooks |
|-------|-----------|-------------|-----------|
| 1 | [**Linear Regression**](Supervised%20Learning/Linear_Regression/lr.md) | Fundamental regression technique | Simple, Multiple, Polynomial |
| 2 | [**Ridge, Lasso & Elastic Net**](Supervised%20Learning/Linear_Regression/lr.md) | Regularized linear models (L1, L2, Combined) | Ridge/Lasso/ElasticNet |
| 3 | [**Logistic Regression**](Supervised%20Learning/Logistic_Regression/lor.md) | Binary and multiclass classification | Binary, Multiclass, ROC-AUC |

### 2. Non-Linear Models (Intermediate)

| Order | Algorithm | Description | Type |
|-------|-----------|-------------|------|
| 4 | [**Support Vector Machines**](Supervised%20Learning/Support%20Vector%20Machines%20(SVM)/svm.md) | Maximum margin classifier with kernels | Classification & Regression |
| 5 | [**Naive Bayes**](Supervised%20Learning/Naive%20Baye's%20Theorem/nb.md) | Probabilistic classifier based on Bayes' theorem | Classification |
| 6 | [**K-Nearest Neighbors**](Supervised%20Learning/K%20Nearest%20Neighbour/knn.md) | Instance-based learning | Classification & Regression |

### 3. Tree-Based Models (Advanced)

| Order | Algorithm | Description | Type |
|-------|-----------|-------------|------|
| 7 | [**Decision Tree**](Supervised%20Learning/Decision%20Tree/dt.md) | Tree-based model with recursive splitting | Classification & Regression |
| 8 | [**Random Forest**](Supervised%20Learning/Random%20Forest/rf.md) | Ensemble of decision trees (Bagging) | Classification & Regression |
| 9 | [**AdaBoost**](Supervised%20Learning/AdaBoost/ab.md) | Adaptive boosting ensemble | Classification & Regression |
| 10 | [**Gradient Boosting**](Supervised%20Learning/Gradient%20Boosting/gb.md) | Sequential ensemble minimizing loss | Classification & Regression |
| 11 | [**XGBoost**](Supervised%20Learning/Extreme%20Gradient%20Boosting/) | Optimized gradient boosting with regularization | Classification & Regression |

## Unsupervised Learning

### 4. Dimensionality Reduction

| Order | Algorithm | Description |
|-------|-----------|-------------|
| 12 | [**PCA (Principal Component Analysis)**](Unsupervised%20Learning/PCA/pca.md) | Linear dimensionality reduction |

### 5. Clustering Algorithms

| Order | Algorithm | Description |
|-------|-----------|-------------|
| 13 | [**K-Means Clustering**](Unsupervised%20Learning/K%20Means%20Clustering/k-means.md) | Centroid-based clustering for spherical clusters |
| 14 | [**Hierarchical Clustering**](Unsupervised%20Learning/Hierarchical%20Clustering/hc.md) | Tree-based clustering with dendrograms |
| 15 | [**DBSCAN**](Unsupervised%20Learning/DBSCAN%20Clustering/dbscan.md) | Density-based spatial clustering |

### 6. Anomaly Detection

| Order | Algorithm | Description |
|-------|-----------|-------------|
| 16 | [**Isolation Forest**](Unsupervised%20Learning/Anomaly%20Detection/Isolation%20Forest/if.md) | Tree-based anomaly detection |
| 17 | [**Local Outlier Factor**](Unsupervised%20Learning/Anomaly%20Detection/Local%20Outlier%20Anomaly/lof.md) | Density-based anomaly detection |
| 18 | [**DBSCAN Anomaly Detection**](Unsupervised%20Learning/Anomaly%20Detection/DBSCAN%20Anomaly%20Detection/dbs_anomaly.md) | Density-based outlier identification |

## Quick Start

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- Required packages listed in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/Abhinavexists/ML-Algorithms.git
cd ML-Algorithms

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Usage

```python
# Example: Linear Regression from scratch
from Supervised Learning.Linear_Regression.linear_regression import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Repository Structure

```text
ML-Algorithms/
├── Supervised Learning/
│   ├── Linear_Regression/          # 1. Foundation
│   ├── Logistic_Regression/        # 2. Classification basics
│   ├── Support Vector Machines/    # 3. Non-linear models
│   ├── Naive Baye's Theorem/       # 4. Probabilistic models
│   ├── K Nearest Neighbour/        # 5. Instance-based learning
│   ├── Decision Tree/              # 6. Tree models
│   ├── Random Forest/              # 7. Ensemble (Bagging)
│   ├── AdaBoost/                   # 8. Ensemble (Boosting)
│   ├── Gradient Boosting/          # 9. Advanced boosting
│   └── Extreme Gradient Boosting/  # 10. Optimized boosting
├── Unsupervised Learning/
│   ├── PCA/                        # 11. Dimensionality reduction
│   ├── K Means Clustering/         # 12. Clustering
│   ├── Hierarchical Clustering/    # 13. Hierarchical methods
│   ├── DBSCAN Clustering/          # 14. Density-based clustering
│   └── Anomaly Detection/          # 15-17. Outlier detection
├── datasets/                       # Centralized datasets
├── images/                         # Visualizations
└── requirements.txt
```

## Algorithms to Study

The following traditional ML algorithms are not yet implemented but are recommended for a complete understanding:

### Supervised Learning

- **Stacking/Blending** - Meta-ensemble combining multiple model predictions
- **Bayesian Linear Regression** - Probabilistic approach to linear regression

### Unsupervised Learning

- **Gaussian Mixture Models (GMM)** - Probabilistic clustering with soft assignments
- **t-SNE** - Non-linear dimensionality reduction for visualization
- **UMAP** - Uniform Manifold Approximation and Projection
- **Mean Shift Clustering** - Non-parametric clustering using kernel density
- **Spectral Clustering** - Graph-based clustering using eigenvalues
- **OPTICS** - Ordering Points To Identify Clustering Structure
- **Affinity Propagation** - Message-passing clustering algorithm

### Time Series (Traditional ML)

- **ARIMA/SARIMA** - Autoregressive Integrated Moving Average
- **Exponential Smoothing** - Simple, Double, Triple (Holt-Winters)
- **Seasonal Decomposition** - Trend, seasonality, and residual analysis

### Association Rule Learning

- **Apriori Algorithm** - Frequent itemset mining
- **FP-Growth** - Faster frequent pattern mining
- **Eclat Algorithm** - Equivalence Class Transformation

### Other Important Algorithms

- **Self-Organizing Maps (SOM)** - Neural network-based clustering/visualization
- **Learning Vector Quantization (LVQ)** - Prototype-based classification

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
