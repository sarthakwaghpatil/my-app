import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Iris dataset
df = pd.read_csv('C:\\Users\\sarth\\drive\\files\\Iris.csv')  # Replace with your file path
print(df.head())

# Display the first few rows of the dataset
print(df.head())

# Summarize the dataset
print("\nDataset Summary:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Visualizing feature distributions using histograms
df.hist(figsize=(10, 8), bins=15)
plt.tight_layout()
plt.show()

# Boxplots for each feature
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, width=0.5, palette='Set2')
plt.tight_layout()
plt.show()
