import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.tree import export_text
import matplotlib.pyplot as plt

# Create a DataFrame with your dataset
data = {
    'A': [3, 2, 1, 3, 1, 2, 3, 13, 2, 4],
    'B': [0.5, 0.6, 0.1, 0.8, 0.9, 0.5, 0.4, 0.1, 0.1, 0.8],
    'C': [6, 5, 4, 2, 2, 3, 3, 5, 3, 5],
    'E': ['success', 'success', 'success', 'failure', 'failure', 'failure', 'failure', 'success', 'success', 'failure']
}

df = pd.DataFrame(data)

# Convert categorical 'E' column to numeric using LabelEncoder
le = preprocessing.LabelEncoder()
df['E'] = le.fit_transform(df['E'])

# Define features and target
y = df['E']

# Create decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy')
clf = clf.fit(X, y)

# Display the decision tree
tree_rules = export_text(clf, feature_names=['A', 'B', 'C'])
print(tree_rules)

# Plot the decision tree
from sklearn.tree import plot_tree
plt.figure(figsize=(10, 8))
plot_tree(clf, filled=True, feature_names=['A', 'B', 'C'])
plt.show()