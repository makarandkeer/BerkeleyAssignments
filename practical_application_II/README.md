# Used Car Price Prediction Analysis Report

1. Introduction

This report details an analysis conducted to identify and quantify the primary factors influencing used car prices. By understanding these relationships, used car dealers can fine-tune their inventory pricing strategies to optimize sales and profitability. We utilized data from your `vehicles.csv` dataset, focusing on predicting car prices based on following features

    1. Numeric features: odometer, age, cylinders_num
    2. Categorical features: condition, title_status, size


2.  Models Used

To provide a robust understanding of price determinants, we employed three common regression models from the scikit-learn library:

    1.Linear Regression: A standard method that models the relationship between a dependent variable and one or more independent variables by fitting a linear equation to observed data.

    2. Ridge Regression: An extension of linear regression that adds a regularization term to prevent overfitting and improve the model's generalization capabilities, particularly useful when dealing with multicollinearity or a large number of features.

    3. LASSO (Least Absolute Shrinkage and Selection Operator) Regression: Similar to Ridge, LASSO also adds a regularization term, but it has the additional ability to perform feature selection by shrinking the coefficients of less important features to zero.

3. Model Performance

To assess how well each model predicts prices, we used the Root Mean Squared Error (RMSE). RMSE measures the average squared difference between the predicted price and the actual price; a lower MSE indicates a more accurate model.

The RMSE values for our models were:

    Linear Regression: 8615.66

    Ridge Regression: 8615.66

    LASSO Regression: 8616.16

All three models yielded very similar performance in terms of RMSE. Ridge Regression exhibited a marginally lower RMSE, suggesting it might be slightly more robust for predicting used car prices based on the chosen features in this dataset. However, the differences are negligible, indicating that for these two features, the regularization in Ridge and LASSO did not significantly alter the prediction accuracy compared to standard Linear Regression

4. Jupyter Notebook:  practical_application_II.ipynb