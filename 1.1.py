# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'C:\\Users\\sarth\\drive\\files\\salary_data.csv'  # Replace with your file path if necessary
data = pd.read_csv(file_path)

# Step 1: Split the data into features (X) and target (y)
X = data[['YearsExperience']]  # Independent variable
y = data['Salary']             # Dependent variable

# Split the dataset into training (80%) and testing (20%) subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict salaries for both training and testing sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Step 3: Plot scatter plots with regression lines for training and testing datasets
plt.figure(figsize=(12, 6))

# Training data plot
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Actual (Train)')
plt.plot(X_train, y_train_pred, color='red', label='Regression Line (Train)')
plt.title('Training Data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()

# Testing data plot
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='green', label='Actual (Test)')
plt.plot(X_test, y_test_pred, color='orange', label='Regression Line (Test)')
plt.title('Testing Data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()

plt.tight_layout()
plt.show()

# Step 4: Display predicted values for the testing data
print("Predicted Salaries for Testing Data:")
print(y_test_pred)
