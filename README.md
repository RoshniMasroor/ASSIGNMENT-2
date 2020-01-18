# ASSIGNMENT-2
# QUESTION 2: TEMP PREDICTION IN 2017 AND 2016
# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'D:\ML\assignment 2\assignment 2\annual_temp.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Annual Temperature of two Industries')
plt.xlabel('Years')
plt.ylabel('Mean Temperature')
plt.show()

# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('Annual Temperature of two Industries')
plt.xlabel('Years')
plt.ylabel('Mean Temperature')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Annual Temperature of two Industries')
plt.xlabel('Years')
plt.ylabel('Mean Temperature')
plt.show()

# Predicting a new result with Linear Regression
lin_reg.predict([[2017]])
print (lin_reg.predict([[2017]]))
lin_reg.predict([[2016]])       #I've run the whole code separately to predict the temp for 2017 and then for 2016.Here I've written them together
print (lin_reg.predict([[2016]]))
# Predicting a new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform([[2017]]))
print (lin_reg_2.predict(poly_reg.fit_transform([[2017]])))
lin_reg_2.predict(poly_reg.fit_transform([[2016]]))
print (lin_reg_2.predict(poly_reg.fit_transform([[2016]])))
