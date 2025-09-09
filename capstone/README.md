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

#### Next steps
What suggestions do you have for next steps?

#### Outline of project

- [Link to notebook](https://github.com/makarandkeer/BerkeleyAssignments/blob/develop/capstone/heart_disease_indicators.ipynb)



##### Contact and Further Information