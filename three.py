import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target

# Create the Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Apply 5-fold cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5)

# Output the accuracy for each fold
print(f"Accuracy for each fold: {cv_scores}")

# Calculate and print the mean accuracy
mean_accuracy = cv_scores.mean()
print(f"Mean accuracy across 5 folds: {mean_accuracy * 100:.2f}%")
