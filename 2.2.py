import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

# Load the Iris dataset from Kaggle (CSV format)
df = pd.read_csv('C:\\Users\\sarth\\drive\\files\\Iris.csv')  # Replace with your file path
print(df.head())

# Split the dataset into features (X) and target (y)
X = df.drop('Species', axis=1)  # Features (sepal length, width, petal length, width)
y = df['Species']  # Target (species)

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the model on the training data
clf.fit(X_train, y_train)

# Predict the target values on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of Decision Tree Classifier: {accuracy * 100:.2f}%")

# Visualize the decision tree
plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=df['Species'].unique(), fontsize=12)
plt.title("Decision Tree Classifier Visualization")
plt.show()
