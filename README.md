# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. **Import Libraries**:  
   Bring in the necessary libraries.

2. **Load the Dataset**:  
   Load the dataset into your environment.

3. **Data Preprocessing**:  
   Handle any missing data and encode categorical variables as needed.

4. **Define Features and Target**:  
   Split the dataset into features (X) and the target variable (y).

5. **Split Data**:  
   Divide the dataset into training and testing sets.

6. **Build Multiple Linear Regression Model**:  
   Initialize and create a multiple linear regression model.

7. **Train the Model**:  
   Fit the model to the training data.

8. **Evaluate Performance**:  
   Assess the model's performance using cross-validation.

9. **Display Model Parameters**:  
   Output the model’s coefficients and intercept.

10. **Make Predictions & Compare**:  
    Predict outcomes and compare them to the actual values. 

## Program:
```python
'''
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: Sanjushri A
RegisterNumber: 21223040187
'''
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: Vishwaraj G.
RegisterNumber: 212223220125
*/
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = 'encoded_car_data.csv'
df = pd.read_csv(file_path)

# Select relevant features and target variable
X = df.drop(columns=['price'])  # All columns except 'price'
y = df['price']  # Target variable

# Split the dataset (not strictly required for cross-validation, but good for validation outside cross-validation)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Multiple Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate on test set
y_pred = model.predict(X_test)
print("Test Set Evaluation:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Cross-Validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')  # 5-fold CV
cv_mse = -cv_scores  # Convert negative MSE to positive
print("\nCross-Validation Results:")
print("MSE for each fold:", cv_mse)
print("Mean MSE:", np.mean(cv_mse))
print("Standard Deviation of MSE:", np.std(cv_mse))

# Cross-Validation R-squared
cv_r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\nR-squared for each fold:", cv_r2_scores)
print("Mean R-squared:", np.mean(cv_r2_scores))
print("Standard Deviation of R-squared:", np.std(cv_r2_scores))

```

## Output:
![image](https://github.com/user-attachments/assets/4ac0fd0b-c2db-4984-b942-d1b897d7db58)



## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
