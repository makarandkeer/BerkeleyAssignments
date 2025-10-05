### Heart disease indicator

**Author** : Makarand Keer

#### Executive summary

#### Research Question
How to accurately identify and reduce the most significant risk factors of heart disease in a large, diverse population, using tools like machine learning to enhance prevention and early detection

#### Rationale
High prevalence of heart disease and its major risk factors among the U.S. population, and the need for improved identification and prevention strategies

#### Data Sources
https://www.cdc.gov/brfss/annual_data/annual_2022.html
The dataset originally comes from the CDC and is a major part of the Behavioral Risk Factor Surveillance System (BRFSS), which conducts annual telephone surveys to collect data on the health status of U.S. residents

Kagel Data : https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

#### Methodology
- For finding outliers IQR and Z-score methods used
- For missing value imputation and transformation based on type of data following methods used
    - For numeric columns used mean
    - For ordinal categorical columns used mapping of values to numeric used
    - For Binary value columns (Yes, No, Null) numeric values used as (1, 0, -1)
    - For other categorical columns, One-hot encoding done


#### Results

- Data is highly imbalance 
- There is no column which is irrelevant and can be dropped

#### Classification Models Comparison

Four classification models were evaluated to predict heart disease indicators:

##### 1. **Logistic Regression**
- **Train Time:** 17.338 seconds
- **Train Accuracy:** 94.61%
- **Test Accuracy:** 94.64%
- **Pros:**
  - Fast training time compared to SVM
  - Good balance between precision (59%) and recall (20%) for positive class
  - Interpretable coefficients for feature importance
  - Handles large datasets efficiently with 'saga' solver
- **Cons:**
  - Low recall (20%) for heart disease cases (Class 1)
  - Assumes linear relationship between features and target
  - May underperform with complex non-linear patterns

##### 2. **K-Nearest Neighbors (KNN)**
- **Train Time:** 0.133 seconds (fastest)
- **Train Accuracy:** 94.68%
- **Test Accuracy:** 94.10%
- **Pros:**
  - Extremely fast training time
  - No assumptions about data distribution
  - Simple and intuitive algorithm
- **Cons:**
  - Very poor recall (3%) for heart disease cases
  - Low precision (32%) for positive class
  - Computationally expensive during prediction phase
  - Sensitive to feature scaling and curse of dimensionality

##### 3. **Decision Tree**
- **Train Time:** 3.847 seconds
- **Train Accuracy:** 100.00% (overfitting)
- **Test Accuracy:** 91.69%
- **Pros:**
  - Best recall (31%) among all models for heart disease cases
  - Easy to interpret and visualize
  - Handles non-linear relationships well
  - No need for feature scaling
- **Cons:**
  - Clear signs of overfitting (100% train accuracy vs 91.69% test accuracy)
  - Lowest test accuracy among all models
  - Low precision (29%) for positive class
  - Prone to high variance

##### 4. **Support Vector Machine (SVM)**
- **Train Time:** 7254.288 seconds (slowest by far)
- **Train Accuracy:** 94.31%
- **Test Accuracy:** 94.31%
- **Pros:**
  - Consistent performance between train and test sets
  - Effective in high-dimensional spaces
  - Memory efficient (uses support vectors)
- **Cons:**
  - Extremely long training time (over 2 hours)
  - Failed to predict any positive cases (0% recall for Class 1)
  - Not suitable for large datasets without optimization
  - Difficult to interpret

##### Key Findings:
- **Baseline Accuracy:** 94.31% (predicting majority class)
- **Best Overall Model:** Logistic Regression - offers the best balance of accuracy (94.64%), training time (17.3s), and ability to detect positive cases
- **Critical Issue:** All models struggle with the highly imbalanced dataset (Class 0: 81,358 vs Class 1: 4,912)
- **Recommendation:** Decision Tree shows promise for recall but needs regularization to prevent overfitting. Consider techniques like SMOTE, class weighting, or ensemble methods to improve minority class detection

#### Next steps
1. Address class imbalance using SMOTE, ADASYN, or class weighting techniques
2. Implement ensemble methods (Random Forest, XGBoost, Gradient Boosting)
3. Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV
4. Apply feature selection to reduce dimensionality (93 features after encoding)
5. Use cross-validation for more robust model evaluation
6. Optimize for recall/F1-score rather than accuracy given the imbalanced data
7. Consider cost-sensitive learning where false negatives are more costly

#### Outline of project

- [Link to notebook](https://github.com/makarandkeer/BerkeleyAssignments/blob/develop/capstone/heart_disease_indicators.ipynb)



##### Contact and Further Information