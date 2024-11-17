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
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load the dataset
file_path = '/mnt/data/encoded_car_data (1).csv'
car_data = pd.read_csv(file_path)

# Prepare features (X) and target (y)
X = car_data.drop(columns=['price'])  # All columns except 'price'
y = car_data['price']  # Target variable

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Evaluate the model using 5-fold cross-validation on the training set
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
cv_mse = -cv_scores.mean()  # Convert to positive MSE
cv_rmse = np.sqrt(cv_mse)  # Calculate Root Mean Squared Error (RMSE)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate performance metrics on the test set
test_mse = mean_squared_error(y_test, y_pred)
test_r2 = r2_score(y_test, y_pred)

# Print the results
print("Cross-Validation RMSE: {:.2f}".format(cv_rmse))
print("Test MSE: {:.2f}".format(test_mse))
print("Test R² Score: {:.3f}".format(test_r2))

```

## Output:
![image](https://github.com/user-attachments/assets/4ac0fd0b-c2db-4984-b942-d1b897d7db58)



## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
