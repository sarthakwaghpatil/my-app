import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset from a CSV file
# Ensure the CSV file is in the same directory or provide the correct path
# Example: iris.csv should have columns like 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', and 'species'
iris = pd.read_csv('C:\\Users\\sarth\\drive\\files\\Iris.csv')

# Display the first few rows of the dataset to check the format
print(iris.head())

# Features (X) and Target (y)
X = iris.drop('species', axis=1)  # Drop the target column (species)
y = iris['species']  # The target column

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
base_model = DecisionTreeClassifier(random_state=42)

# Apply the Bagging technique using the Decision Tree Classifier as the base model
bagging_model = BaggingClassifier(estimator=base_model, n_estimators=50, random_state=42)

# Train the Bagging model
bagging_model.fit(X_train, y_train)

# Predict the target values on the test set
y_pred = bagging_model.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Bagging Classifier: {accuracy * 100:.2f}%")
