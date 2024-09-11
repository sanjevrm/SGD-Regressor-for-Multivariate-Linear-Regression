# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Load and Prepare Dataset
2.Split Dataset
3.Scale Features and Target
4.Train the Model
5.Predict and Evaluate
```
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:JAGADEESH J
RegisterNumber:212223110015


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
*/
```
## Using The Inbuilt Dataset:
```
data=fetch_california_housing()
print(data)
```
## Output:
![image](https://github.com/user-attachments/assets/ddb20c3e-7d75-4929-9b73-ab293a05f808)

## Changing from Array to Rows and Columns:
```
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
df.head()
```
## Output:
![image](https://github.com/user-attachments/assets/043c3c6b-f37b-450a-a73f-90f38eaaab39)

## Information of the Dataset:
```
df.info()
```
## Output:
![image](https://github.com/user-attachments/assets/c6419e6d-c557-455c-87ab-f01ef7fb01c5)

## Spliting for Output:
```
x=df.drop(columns=['traget','AveOccup'])
x.info()
Y=df[['traget','AveOccup']]
Y.info()
```
## Output:
![image](https://github.com/user-attachments/assets/350296e7-bfc1-456f-96c2-15455ceb1e29)
![image](https://github.com/user-attachments/assets/80a39a4d-f578-4f53-b769-33c8fa1700e7)

## Training and Testing the Models:
```
x_train,x_test,y_train,y_test=train_test_split(x,Y,test_size=0.2,random_state=1)
x.head()
```
## Output:
![image](https://github.com/user-attachments/assets/32a15d7f-8569-4cd0-ba63-8ebfb207d5fc)

## StandardScaler:
```
scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
y_train=scaler_y.fit_transform(y_train)
x_test=scaler_x.transform(x_test)
y_test=scaler_y.transform(y_test)
print(x_train)
```
## Output:
![image](https://github.com/user-attachments/assets/5da620a9-459f-43c2-8cb8-10acc058cfce)

## PREDICTION:
```
sdg=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sdg=MultiOutputRegressor(sdg)
multi_output_sdg.fit(x_train,y_train)
Y_pred=multi_output_sdg.predict(x_test)
Y_pred=scaler_y.inverse_transform(Y_pred)
Y_test=scaler_y.inverse_transform(y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
```
## Output:
![image](https://github.com/user-attachments/assets/8c11918e-effc-4e35-bddf-17dd0a8e5ace)

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
