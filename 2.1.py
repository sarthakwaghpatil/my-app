import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv('\files\\Iris.csv')
print(df.head())


print(df.head())

print("\nDataset Summary:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

df.hist(figsize=(10, 8), bins=15)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 8))
sns.boxplot(data=df, width=0.5, palette='Set2')
plt.tight_layout()
plt.show()
