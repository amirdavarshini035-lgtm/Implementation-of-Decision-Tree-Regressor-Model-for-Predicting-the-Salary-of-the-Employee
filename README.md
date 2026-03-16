# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the libraries and read the data frame using pandas.
2.Calculate the null values present in the dataset and apply label encoder.
3.Determine test and training data set and apply decison tree regression in dataset.
4.Calculate Mean square error,data prediction and r2.  

## Program:
```
import pandas as pd

# Load dataset
data = pd.read_csv("Salary.csv")

# Check dataset
print(data.head())
print(data.info())
print(data.isnull().sum())

# Convert Position (text) to numeric
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])

print(data.head())

# Features and Target
x = data[["Position","Level"]]
y = data["Salary"]

# Split dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train, y_train)

# Prediction
y_pred = dt.predict(x_test)

# Evaluation
from sklearn import metrics
mse = metrics.mean_squared_error(y_test, y_pred)
print("MSE:", mse)

r2 = metrics.r2_score(y_test, y_pred)
print("R2 Score:", r2)

# Predict new value
print("Predicted Salary:", dt.predict([[5,6]]))

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: AMIRDAVARSHINI D
RegisterNumber:212225230013  

```

## Output:
<img src="https://img.sanishtech.com/u/c0d9f69274009d9243f350b5524ef71d.png" alt="Screenshot 2026-03-16 113031" width="1406" height="677" loading="lazy" style="max-width:100%;height:auto;">

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
