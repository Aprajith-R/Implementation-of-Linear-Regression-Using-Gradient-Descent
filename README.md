# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. :start
2. Import the required library and read the dataframe.
3. Write a function computeCost to generate the cost function.
4. Perform iterations og gradient steps with learning rate.
5. Plot the Cost function using Gradient Descent and generate the required graph.
6. stop

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Aprajith R
RegisterNumber: 212222080006
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
  x = np.c_[np.ones(len(x1)),x1]
  theta = np.zeros(x.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions = (x).dot(theta).reshape(-1,1)
    errors = (predictions - y).reshape(-1,1)
    theta = learning_rate * (1/len(x1))*x.T.dot(errors)
  return theta
data=pd.read_csv("50_Startups.csv",header=None)
print(data.head())
x = (data.iloc[1:, :-2].values)
print(x)
x1=x.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
print(y)
x1_Scaled = scaler.fit_transform(x1)
y1_Scaled = scaler.fit_transform(y)
print(x1_Scaled)
print(y1_Scaled)
theta = linear_regression(x1_Scaled, y1_Scaled )
new_data = np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled = scaler.fit_transform(new_data)
prediction = np.dot(np.append(1, new_Scaled),theta)
prediction = prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value:{pre}")
*/
```

## Output:
![Screenshot 2024-05-14 191405](https://github.com/Aprajith-R/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161153978/4a24f8ec-0eec-4c73-abc1-2e9316dc643b)

![Screenshot 2024-05-14 191416](https://github.com/Aprajith-R/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161153978/ee3884bb-01e8-41b1-b7b3-72c56aba0610)

![Screenshot 2024-05-14 191425](https://github.com/Aprajith-R/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161153978/b0ce4b77-1ff2-4e53-8edc-a063ac865303)

![Screenshot 2024-05-14 191433](https://github.com/Aprajith-R/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161153978/fa1286ce-ff9e-40e7-b96b-903a928be2c6)

![Screenshot 2024-05-14 191441](https://github.com/Aprajith-R/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161153978/39c100df-5567-4063-87a5-e3b87e885137)

![Screenshot 2024-05-14 191448](https://github.com/Aprajith-R/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/161153978/8b3caffd-9e51-4f46-b23f-5aa422612fc3)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
