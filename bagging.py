import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

iris = pd.read_csv('files\\Iris.csv')

print(iris.head())

X = iris.drop('species', axis=1)  # Drop the target column (species)
y = iris['species']  # The target column

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_model = DecisionTreeClassifier(random_state=42)

bagging_model = BaggingClassifier(estimator=base_model, n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)
y_pred = bagging_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"acuracy of bagging classifier: {accuracy * 100:.2f}%")
