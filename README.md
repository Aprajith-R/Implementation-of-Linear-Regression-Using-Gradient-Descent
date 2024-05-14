# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.

2. Write a function computeCost to generate the cost function.

3. Perform iterations og gradient steps with learning rate.

4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Ajay Joshua . M
RegisterNumber: 212222080004

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
print(X)
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
  
/*

```

## Output:
![Screenshot 2024-03-15 123753](https://github.com/Ajay-Joshua-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/160995404/2c62ae96-ed8a-45f4-8d94-b4b8fb2a203b)

![image](https://github.com/Ajay-Joshua-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/160995404/d4dde488-b2ad-4133-9b8e-708ceaf8457b)

![image](https://github.com/Ajay-Joshua-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/160995404/db3b1a48-b157-4a90-bde9-9262d1ab62d2)

![image](https://github.com/Ajay-Joshua-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/160995404/7386cfb8-285d-4dc8-a3e4-ccd5be4de16b)

![image](https://github.com/Ajay-Joshua-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/160995404/91ec5761-3a30-4b10-93f0-72813db09b82)

![image](https://github.com/Ajay-Joshua-M/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/160995404/28c11e18-dc14-48a7-bd27-34b01f7c41de)



## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
