import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv('\files\\Iris.csv') 
print(df.head())


X = df.drop('Species', axis=1)  
y = df['Species'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accracyy of decisiontree classifier: {accuracy * 100:.2f}%")

plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=X.columns, class_names=df['Species'].unique(), fontsize=12)
plt.title("Decision Tree Classifier Visualization")
plt.show()
