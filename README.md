# Ames Housing Price Prediction (Machine Learning Project)

This project applies a full machine learning pipeline to predict house prices using the Ames Housing dataset. It includes exploratory data analysis (EDA), preprocessing, feature engineering, model training, and evaluation of multiple regression algorithms.

## Overview

The goal of this project is to build a predictive model that estimates house prices based on structured tabular features. The workflow follows a standard data science pipeline from raw data exploration to final model comparison.

## Dataset

The project is based on the  **Ames Housing dataset** , which contains detailed information about residential properties, including:

* Physical property attributes
* Location and neighborhood features
* Size, quality, and construction details

## Methodology

### 1. Exploratory Data Analysis (EDA)

* Analyzed key variables such as `LotFrontage` and basement-related features
* Investigated missing values, outliers, and invalid entries

### 2. Data Preprocessing

* Removed ID-like columns that are not useful for prediction
* Imputed missing values using strategies such as:
  * Neighborhood-based median imputation for `LotArea`
  * Feature-specific imputation for missing attributes
* Handled missing categorical and numerical data consistently

### 3. Feature Engineering

* Created new informative variables such as `HouseAge`

### 4. Skewness Handling

* Applied log transformation to reduce skewness in highly distributed features
* Preserved outliers while reducing their impact on model training

### 5. Encoding

* Converted categorical variables into numerical format using appropriate encoding techniques (e.g., one-hot encoding depending on feature type)

## Models Evaluated

Multiple regression models were trained and compared:

* Linear Regression
* Ridge Regression
* Lasso Regression
* ElasticNet
* Random Forest
* Gradient Boosting
* XGBoost
* Neural Network (MLP)

## Baseline Model

A simple baseline model was implemented by predicting the mean target value for all samples. This provides a reference point for evaluating the performance of more complex models.

All trained models significantly outperformed the baseline, confirming that the models successfully learned meaningful patterns from the data.

## Results (Log Space)

| Model (dollars)         | MAE   | RMSE  |
| ----------------------- | ----- | ----- |
| XGBoost                 | 14047 | 23004 |
| Gradient Boosting       | 14218 | 26121 |
| Linear Regression       | 15989 | 27049 |
| Ridge                   | 16049 | 30525 |
| Random Forest           | 16096 | 31094 |
| ElasticNet              | 17976 | 33886 |
| Lasso                   | 18532 | 34274 |
| Neural Network          | 27852 | 45257 |
| Baseline MEAN Predictor | 63733 | 92955 |

Key Findings

* Gradient boosting methods (XGBoost, Gradient Boosting) achieved the best performance, confirming their strength for structured tabular data. The best performing model was **XGBoost**, achieving a **MAE** of $14047, which is a 78% improvment over the naive **Mean Predictor** .
* Linear models performed surprisingly well, indicating that feature engineering successfully captured much of the underlying linear structure.
* Neural networks underperformed, highlighting their limitations on smaller tabular datasets without large-scale data or tuning.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost
* Matplotlib

## Conclusion

This project demonstrates a complete machine learning workflow for regression problems, from raw data analysis to model comparison and evaluation. It highlights the importance of feature engineering and shows that well-designed tabular features can make simpler models competitive with more complex approaches.
