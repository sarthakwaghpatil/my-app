import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

iris = load_iris()

X = iris.data  # Features: SepalLength, SepalWidth, PetalLength, PetalWidth
y = iris.target  # Target: Species (Setosa, Versicolor, Virginica)

clf = DecisionTreeClassifier(random_state=42)

cv_scores = cross_val_score(clf, X, y, cv=5)

print(f"Accuracy for each fold: {cv_scores}")

mean_accuracy = cv_scores.mean()
print(f"Mean accuracy across 5 folds: {mean_accuracy * 100:.2f}%")