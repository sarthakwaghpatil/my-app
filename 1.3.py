import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset (adjust file path as needed)
df = pd.read_csv('C:\\Users\\sarth\\drive\\files\\user_data.csv')  # Replace with your file path
print(df.head())

# Preprocess the data (Assuming 'Feature1', 'Feature2', and 'Target' are the columns in the dataset)
X = df[['EstimatedSalary', 'Age']].values  # Replace with your feature column names
y = df['Purchased'].values  # Replace with your target column

# Split the data into training and testing subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the logistic regression model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Evaluate the model
y_pred = log_reg.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


# Function to plot decision boundary
def plot_decision_boundary(X, y, model, title="Logistic Regression Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.75, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='coolwarm', marker='o')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


# Plot decision boundary for training data
plot_decision_boundary(X_train, y_train, log_reg, "Training Data: Logistic Regression Decision Boundary")

# Plot decision boundary for testing data
plot_decision_boundary(X_test, y_test, log_reg, "Testing Data: Logistic Regression Decision Boundary")
