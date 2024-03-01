from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing

# Example dataset
X = [
    ['Sunny', 'Hot', 'High', 'Weak'],
    ['Sunny', 'Hot', 'High', 'Strong'],
    ['Overcast', 'Hot', 'High', 'Weak'],
    ['Rain', 'Mild', 'High', 'Weak'],
    ['Rain', 'Cool', 'Normal', 'Weak'],
    ['Rain', 'Cool', 'Normal', 'Strong'],
    ['Overcast', 'Cool', 'Normal', 'Strong'],
    ['Sunny', 'Mild', 'High', 'Weak'],
    ['Sunny', 'Cool', 'Normal', 'Weak'],
    ['Rain', 'Mild', 'Normal', 'Weak'],
    ['Sunny', 'Mild', 'Normal', 'Strong'],
    ['Overcast', 'Mild', 'High', 'Strong'],
    ['Overcast', 'Hot', 'Normal', 'Weak'],
    ['Rain', 'Mild', 'High', 'Strong']
]
y = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

# Encode categorical features
encoder = preprocessing.OrdinalEncoder()
X_encoded = encoder.fit_transform(X)

# Feature names
feature_names = ['Outlook', 'Temperature', 'Humidity', 'Wind']

# Create and train the DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X_encoded, y)

# Print the decision tree
from sklearn.tree import export_text
tree_rules = export_text(clf, feature_names=feature_names)
print(tree_rules)