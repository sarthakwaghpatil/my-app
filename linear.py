
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


file_path = '\files\salary_data.csv'
data = pd.read_csv(file_path)

//splitting daata by me
X = data[['YearsExperience']]
y = data['Salary']             

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color='blue', label='Actual (Train)')
plt.plot(X_train, y_train_pred, color='red', label='Regression Line (Train)')
plt.title('Training Data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()


plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color='green', label='Actual (Test)')
plt.plot(X_test, y_test_pred, color='orange', label='Regression Line (Test)')
plt.title('Testing Data')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()

plt.tight_layout()
plt.show()


print("Predicted Salaries for Testing Data:")
print(y_test_pred)
