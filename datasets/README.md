# Datasets

This directory contains all datasets used across the ML-Algorithms repository. Centralizing datasets eliminates duplication and ensures consistency across all implementations.

## Available Datasets

### 1. cardekho_imputated.csv

- **Size**: ~1.3 MB
- **Used in**: AdaBoost, XGBoost, Gradient Boosting, Random Forest (Regression)
- **Description**: Car price dataset with imputed missing values
- **Features**: Car specifications and pricing information
- **Target**: Price prediction (regression task)

### 2. Travel.csv

- **Size**: ~458 KB
- **Used in**: AdaBoost, XGBoost, Gradient Boosting, Random Forest (Classification)
- **Description**: Travel package purchase prediction dataset
- **Target**: Binary classification (package purchased or not)

### 3. Algerian_forest_fires_cleaned_dataset.csv

- **Size**: ~15 KB
- **Used in**: Linear Regression
- **Description**: Cleaned dataset of Algerian forest fires
- **Features**: Weather and environmental factors
- **Target**: Fire prediction/severity

### 4. Algerian_forest_fires_dataset_UPDATE.csv

- **Size**: ~15 KB
- **Used in**: Linear Regression
- **Description**: Updated version of Algerian forest fires dataset
- **Note**: Contains additional or corrected data compared to cleaned version

### 5. Economic_Index.csv

- **Size**: ~633 bytes
- **Used in**: Linear Regression
- **Description**: Economic indicators dataset
- **Features**: Various economic metrics

### 6. Height_Weight.csv

- **Size**: ~201 bytes
- **Used in**: Linear Regression
- **Description**: Simple dataset relating height and weight
- **Purpose**: Demonstrating simple linear regression

### 7. healthcare.csv

- **Size**: ~9.2 KB
- **Used in**: Isolation Forest (Anomaly Detection)
- **Description**: Healthcare data for anomaly detection
- **Purpose**: Identifying outliers in healthcare metrics

## Usage in Code

All notebooks and scripts should reference datasets using relative paths from their location:

```python
# For notebooks in Supervised Learning/Algorithm/
df = pd.read_csv('../../datasets/filename.csv')

# For notebooks in Unsupervised Learning/Algorithm/SubCategory/
df = pd.read_csv('../../../datasets/filename.csv')
```

## Data Processing

- **Raw Data Only**: This directory contains only raw/original datasets
- **Preprocessing**: Each algorithm handles its own data preprocessing
- **No Modified Data**: Processed or transformed data should not be stored here

## Adding New Datasets

When adding new datasets:

1. Place the raw dataset file in this directory
2. Update this README with dataset information
3. Document the dataset source and license if applicable
4. Update notebook references to use `../../datasets/` or `../../../datasets/`
