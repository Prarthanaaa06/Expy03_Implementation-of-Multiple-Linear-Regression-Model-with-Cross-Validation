# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: Prarthana D
RegisterNumber:  212225230213
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv('CarPrice_Assignment.csv')

#Simple preprocessing 
# axis =1 means column;  axis =0 means row
data = df.drop(['car_ID','CarName'], axis=1)
data = pd.get_dummies(data, drop_first=True)
#Split data
X = data.drop("price",axis=1)
y = data['price']
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2, random_state=42)

#Create and train model
model = LinearRegression()
model.fit(X_train,y_train)

#Evaluate models
print("Name: PRARTHANA D")
print("Reg. No: 212225230213")
print("\n=== CROSS-VALIDATION ===")
cv_scores = cross_val_score(model,X,y,cv=5)
print("Fold R² scores: ", [f"{score:.4f}" for score in cv_scores])
print(f"Average R²: {cv_scores.mean():.4f}")

#Test set evaluvation
y_pred = model.predict(X_test)
print("\n=== TEST SET PERFORMANCE ===")
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"MAE: {mean_absolute_error(y_test,y_pred):.2f}")
print(f"R²: {r2_score(y_test,y_pred):.4f}")

#Visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(),y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual VS Predicted Car Prices")
plt.show()
```

## Output:
<img width="1055" height="284" alt="Screenshot 2026-02-20 090243" src="https://github.com/user-attachments/assets/c8acc21c-118d-470f-bb54-55f9f42492ab" />

<img width="1265" height="760" alt="image" src="https://github.com/user-attachments/assets/78ca8d7c-dfba-4a2d-8909-937c41e4bbef" />


## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
