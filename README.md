# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
1.Load the dataset Use datasets.fetch_california_housing() to import the California housing dataset.

2.Split the data Apply train_test_split() to divide the dataset into training and testing sets (e.g., 60% train, 40% test).

3.Train the model Create a LinearRegression() object and fit it with the training data (X_train, y_train).

4.Evaluate the model Print the regression coefficients (reg.coef_) and the variance score (reg.score(X_test, y_test)).

5.Visualize residuals Plot residual errors for both training and testing predictions using matplotlib scatter plots, with a horizontal line at zero for reference.

## Program:
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# Load California housing dataset
housing = datasets.fetch_california_housing()
X = housing.data
y = housing.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

# Train linear regression model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)

# Print coefficients and score
print("Coefficients:", reg.coef_)
print("Variance score:", reg.score(X_test, y_test))

# Plot residual errors
plt.style.use('fivethirtyeight')
plt.scatter(reg.predict(X_train), reg.predict(X_train) - y_train, color="green", s=10, label='Train data')
plt.scatter(reg.predict(X_test), reg.predict(X_test) - y_test, color="blue", s=10, label='Test data')
plt.hlines(y=0, xmin=0, xmax=5, linewidth=2)
plt.legend(loc='upper right')
plt.title("Residual Errors")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")
plt.show()


```
## Output:

![exp10 ai](https://github.com/user-attachments/assets/d8a8b2f6-f35c-41c4-a5cc-61fb5bcb0e0e)

![WhatsApp Image 2025-12-18 at 10 21 43 AM](https://github.com/user-attachments/assets/3c85375a-e964-46f8-91b6-db4c1194ae0a)

## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
