# 🏥 Decision Tree vs. Naïve Bayes Classifiers for Cancer Data Classification

## 📚 Table of Contents
1. 🧐 Project Overview
2. 🤔 What We Expect from This Project
3. 👨‍💻 Code and Resources Used
4. 📕 Contents
5. 🧼 Data Preprocessing
   - Handling Missing Values
   - Feature Engineering
6. 📊 Exploratory Data Analysis (EDA)
   - Feature Distribution
   - Correlation Analysis
   - Class Distribution
7. 🤖 Machine Learning Models
   - Decision Tree Classifier
   - Naïve Bayes Classifier
   - Model Performance Evaluation
8. 📈 Results & Conclusion
9. 🎯 Summary

## 🧐 Project Overview
- Perform a comparative analysis of Decision Tree and Naïve Bayes classifiers using a cancer dataset from Kaggle.
- Evaluate classification performance in terms of **accuracy, speed, and efficiency**.
- Conduct feature engineering and statistical analysis to improve model understanding and interpretability.
- Implement various preprocessing techniques such as handling missing values, feature scaling, and encoding categorical variables.
- Compare classification results using standard evaluation metrics such as **accuracy, precision, recall, and F1-score**.

## 🤔 What We Expect from This Project
- **Data preprocessing**, including handling missing values, feature selection, and normalization.
- **Exploratory Data Analysis (EDA)** to understand the dataset and identify important patterns.
- **Implementation of Decision Tree and Naïve Bayes classifiers** and comparison of their performance.
- **Hyperparameter tuning** to optimize both models.
- **Statistical analysis** to evaluate the significance of differences between the models.

## 👨‍💻 Code and Resources Used
**Requirements:** Python, Jupyter Notebook, Scikit-Learn, Pandas, Matplotlib, Seaborn

**Tools Used:** Kaggle, Jupyter Notebook for data processing and visualization

**Packages:** numpy, pandas, seaborn, matplotlib, scikit-learn

## 📕 Contents
- Part 1: Data Preprocessing and Cleaning
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR-LINK-HERE)
- Part 2: Exploratory Data Analysis
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR-LINK-HERE)
- Part 3: Model Implementation and Comparison
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/YOUR-LINK-HERE)

# 🧼 Data Preprocessing

## Handling Missing Values
- Removed records with excessive missing values.
- Applied **mean/mode imputation** for numerical and categorical missing values.
- Standardized numerical features to improve model performance.

## Feature Engineering
- Selected **most relevant features** using correlation analysis.
- Created new features to enhance predictive power.

Example:
```python
# Handling missing values using mean imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
data[['feature1', 'feature2']] = imputer.fit_transform(data[['feature1', 'feature2']])
```

# 📊 Exploratory Data Analysis (EDA)
### Feature Distribution
Understanding feature distributions using histograms and density plots.

### Correlation Analysis
Heatmap visualization to examine correlations among features.

### Class Distribution
Bar plots to explore the balance between different classes in the dataset.

# 🤖 Machine Learning Models

## Decision Tree Classifier
Implemented a **Decision Tree** model and tuned hyperparameters such as max depth and minimum samples per leaf.

```python
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(max_depth=5, random_state=42)
model_dt.fit(X_train, y_train)
```

## Naïve Bayes Classifier
Implemented a **Gaussian Naïve Bayes** model optimized for categorical and continuous data.

```python
from sklearn.naive_bayes import GaussianNB
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
```

## Model Performance Evaluation
### Classification Metrics
- **Accuracy**
- **Precision & Recall**
- **F1-Score**
- **Confusion Matrix**

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, model_dt.predict(X_test)))
print(classification_report(y_test, model_nb.predict(X_test)))
```

### Model Comparison
#### Accuracy Comparison
Bar plots to compare the accuracy of Decision Tree and Naïve Bayes.

#### Execution Speed
Comparison of execution time for both models to evaluate efficiency.

# 📈 Results & Conclusion
- **Decision Tree achieved higher accuracy** but required more computation time.
- **Naïve Bayes performed faster** but had slightly lower accuracy.
- **Future Work:** Optimize feature selection and explore ensemble techniques for better performance.

# 🎯 Summary
This project compared **Decision Tree** and **Naïve Bayes** classifiers for cancer data classification. Future improvements could include hyperparameter tuning with GridSearchCV and feature selection techniques to enhance model performance.

If you liked this work, feel free to ⭐ the repository. Open to collaborations!

