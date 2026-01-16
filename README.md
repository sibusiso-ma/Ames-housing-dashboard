Ames Housing dashboard 

I built a streamlit dashboard for Housing data
from Kaggle.

The dashboard has the following preprocessing steps:
1. Descriptive statistics
2. Exploratory data analysis
 
The exploratory data analysis has the following:
- Histogram Distribution of numerical features
- Scatter Plot that shows relationship between any two numerical features
- Correlation matrix that shows which numerical features have a strong correlation with the house price
3. Pipelines:
I used pipeline to clean data and for data imputation to avoid data leakage.
The pipeline has the preprocessing steps for numerical features:

- Data imputation using the mean of dataset
- Standard scaling to normalize the dataset
  
It has the following preprocessing steps for categorical features:
- Data imputation using the mode
- OneHot encoding

The dashboard allows you to select various Regression models
for the user to compare between the various models.
The following models can be selected:

1. Linear Regression
2. Lasso Regression
3. Polynomial Regression
4. Random Forest

   
The following metrics are used to test accuracy of model:
1. MAE (Mean absolute error)
2. RMSE (Root Mean Square Error)
3.R²

The most accurate model is the Random Forest model.
For each model the user can see the feature importance of each feature used for the model.
This can allow user to understand which features have the most impact on house prices.




