# Naive Bayes Algorithm

## Theory

Naive Bayes is a probabilistic machine learning algorithm based on Bayes' Theorem with an assumption of independence among predictors. It's called "naive" because it assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.

### Bayes' Theorem

The fundamental equation of Naive Bayes is:

$$P(A|B) = \frac{P(B|A) \times P(A)}{P(B)}$$

Where:

- P(A|B) is the posterior probability: the probability of hypothesis A being true given that B is true
- P(B|A) is the likelihood: the probability of B being true given that A is true
- P(A) is the prior probability: the probability of hypothesis A being true
- P(B) is the marginal probability: the probability of B being true

### Types of Naive Bayes

1. **Gaussian Naive Bayes**: Used when features follow a normal distribution
2. **Multinomial Naive Bayes**: Used for discrete counts (e.g., word counts in text)
3. **Bernoulli Naive Bayes**: Used for binary/boolean features

### Advantages

- Fast and easy to implement
- Works well with high-dimensional data
- Requires less training data
- Handles both continuous and discrete data
- Not sensitive to irrelevant features

### Disadvantages

- Assumes features are independent (which is rarely true in real-world data)
- Can be outperformed by more complex models
- Requires feature scaling for better performance

## Implementation

### 1. Iris Dataset Implementation

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
x, y = load_iris(return_X_y=True)

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Create and train model
gnb = GaussianNB()
gnb.fit(x_train, y_train)

# Make predictions
y_pred = gnb.predict(x_test)

# Evaluate model
print(accuracy_score(y_pred, y_test))
print(classification_report(y_pred, y_test))
print(confusion_matrix(y_pred, y_test))
```

Results:

- Accuracy: 1.0 (100%)
- Perfect classification for all three iris classes
- Confusion matrix shows no misclassifications

### 2. Tips Dataset Implementation

This implementation demonstrates handling categorical features and more complex data:

```python
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load data
df = sns.load_dataset('tips')

# Prepare features
x = df[['total_bill','tip', 'sex', 'day', 'time', 'size']]
y = df['smoker']

# Feature Encoding
ohe = OneHotEncoder(drop='first', sparse_output=False)
categorical_cols = ['sex','day','time']
ohe_cat = ohe.fit_transform(x[categorical_cols])

# Create encoded dataframe
ohe_df = pd.DataFrame(ohe_cat, columns=ohe.get_feature_names_out(categorical_cols))

# Combine numerical and encoded features
num_val = x.drop(columns=categorical_cols).reset_index(drop=True)
x_final = pd.concat([num_val, ohe_df], axis=1)

# Encode target variable
le = LabelEncoder()
y_final = le.fit_transform(y)

# Train-test split and model training
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.3, random_state=0)
gnb = GaussianNB()
gnb.fit(x_train, y_train)
y_pred = gnb.predict(x_test)
```

Results:

- Accuracy: 0.54 (54%)
- Precision: 0.67 for non-smokers, 0.40 for smokers
- Recall: 0.55 for non-smokers, 0.52 for smokers
- F1-score: 0.60 for non-smokers, 0.45 for smokers

### Key Implementation Steps

1. **Data Preprocessing**:
   - Handling categorical variables using OneHotEncoder
   - Label encoding for target variable
   - Feature scaling (when necessary)

2. **Model Training**:
   - Using GaussianNB for continuous features
   - Train-test split for evaluation
   - Model fitting with training data

3. **Model Evaluation**:
   - Accuracy score
   - Classification report (precision, recall, f1-score)
   - Confusion matrix

### Observations

1. **Iris Dataset**:
   - Perfect classification (100% accuracy)
   - Well-separated classes
   - Gaussian distribution of features

2. **Tips Dataset**:
   - Moderate accuracy (54%)
   - Better at predicting non-smokers than smokers
   - More complex feature relationships
   - Categorical features add complexity
