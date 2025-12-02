# ML-Algorithms

This repository contains implementations of various machine learning algorithms, covering both supervised and unsupervised learning techniques. Each algorithm includes:

## Algorithm Categories

### Supervised Learning

#### Classification Algorithms

| Algorithm | Description |
|-----------|-------------|
| [**AdaBoost**](Supervised%20Learning/AdaBoost/ab.md) | Ensemble method combining weak learners sequentially |
| [**Decision Tree**](Supervised%20Learning/Decision%20Tree/dt.md) | Tree-based model making decisions by splitting data |
| [**Gradient Boosting**](Supervised%20Learning/Gradient%20Boosting/gb.md) | Sequential ensemble method minimizing loss functions |
| [**Extreme Gradient Boosting (XGBoost)**](Supervised%20Learning/Extreme%20Gradient%20Boosting/) | Optimized gradient boosting with regularization |
| [**K-Nearest Neighbors**](Supervised%20Learning/K%20Nearest%20Neighbour/knn.md) | Instance-based learning using distance metrics |
| [**Naive Bayes**](Supervised%20Learning/Naive%20Baye's%20Theorem/nb.md) | Probabilistic classifier based on Bayes' theorem |
| [**Random Forest**](Supervised%20Learning/Random%20Forest/rf.md) | Ensemble of decision trees with bagging |
| [**Support Vector Machines (SVM)**](Supervised%20Learning/Support%20Vector%20Machines%20(SVM)/svm.md) | Maximum margin classifier with kernel tricks |

#### Regression Algorithms

| Algorithm | Description |
|-----------|-------------|
| [**AdaBoost Regressor**](Supervised%20Learning/AdaBoost/) | Ensemble regression using adaptive boosting |
| [**Decision Tree Regressor**](Supervised%20Learning/Decision%20Tree/) | Tree-based regression model |
| [**Gradient Boosting Regressor**](Supervised%20Learning/Gradient%20Boosting/) | Sequential ensemble for regression |
| [**XGBoost Regressor**](Supervised%20Learning/Extreme%20Gradient%20Boosting/) | Optimized gradient boosting regression |
| [**KNN Regressor**](Supervised%20Learning/K%20Nearest%20Neighbour/) | Distance-based regression method |
| [**SVM Regressor**](Supervised%20Learning/Support%20Vector%20Machines%20(SVM)/) | Support vector regression with kernels |

### Unsupervised Learning

#### Clustering Algorithms

| Algorithm | Description |
|-----------|-------------|
| [**DBSCAN**](Unsupervised%20Learning/DBSCAN%20Clustering/dbscan.md) | Density-based spatial clustering for arbitrary shapes |
| [**Hierarchical Clustering**](Unsupervised%20Learning/Hierarchical%20Clustering/hc.md) | Tree-based clustering with dendrograms |
| [**K-Means**](Unsupervised%20Learning/K%20Means%20Clustering/k-means.md) | Centroid-based clustering for spherical clusters |

#### Dimensionality Reduction

| Algorithm | Description |
|-----------|-------------|
| [**Principal Component Analysis (PCA)**](Unsupervised%20Learning/PCA/pca.md) | Linear dimensionality reduction technique |

#### Anomaly Detection

| Algorithm | Description |
|-----------|-------------|
| [**Isolation Forest**](Unsupervised%20Learning/Anomaly%20Detection/Isolation%20Forest/if.md) | Tree-based anomaly detection |
| [**Local Outlier Factor**](Unsupervised%20Learning/Anomaly%20Detection/Local%20Outlier%20Anomaly/lof.md) | Density-based anomaly detection |
| [**DBSCAN for Anomaly Detection**](Unsupervised%20Learning/Anomaly%20Detection/DBSCAN%20Anomaly%20Detection/dbs_anomaly.md) | Density-based outlier identification |

## Quick Start

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

```bash
# Clone the repository
git clone https://github.com/Abhinavexists/ML-Algorithms.git
cd ML-Algorithms

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```text
ML-Algorithms/
├── Supervised Learning/
│   ├── AdaBoost/
│   ├── Decision Tree/
│   ├── Gradient Boosting/
│   ├── Extreme Gradient Boosting/
│   ├── K Nearest Neighbour/
│   ├── Naive Baye's Theorem/
│   ├── Random Forest/
│   └── Support Vector Machines (SVM)/
├── Unsupervised Learning/
│   ├── Anomaly Detection/
│   ├── DBSCAN Clustering/
│   ├── Hierarchical Clustering/
│   ├── K Means Clustering/
│   └── PCA/
├── images/                 # Algorithm visualizations
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Contributing

Contributions are welcome! Please feel free to:

- Report bugs and issues
- Suggest new algorithms or improvements
- Submit pull requests with enhancements
- Improve documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
