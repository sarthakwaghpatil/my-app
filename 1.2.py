import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('C:\\Users\\sarth\\drive\\files\\Position_Salaries.csv')  # Replace with your file path
print(df.head())

# Select independent and dependent variables
X = df.iloc[:, 1:2].values  # Position Levels
y = df.iloc[:, 2].values    # Salary

# Split dataset into training and testing subsets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Polynomial regression model (degree = 4)
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X_train)

# Train the model
poly_model = LinearRegression()
poly_model.fit(X_poly, y_train)

# Make predictions on the test set
X_test_poly = poly_reg.transform(X_test)
y_pred = poly_model.predict(X_test_poly)

# Display predictions
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(predictions)

# Plotting the polynomial regression curve for training data
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, poly_model.predict(poly_reg.transform(X_train)), color='red')
plt.title('Polynomial Regression (Training Data)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Plotting the polynomial regression curve for testing data
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, poly_model.predict(poly_reg.transform(X_train)), color='red')
plt.title('Polynomial Regression (Testing Data)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Calculate the predicted values for both datasets
y_train_pred = poly_model.predict(poly_reg.transform(X_train))
print("Predicted Salary for Training Data:", y_train_pred)

y_test_pred = y_pred
print("Predicted Salary for Testing Data:", y_test_pred)
