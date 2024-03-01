from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

# Example dataset
# dataset = [
#     ['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
#     ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
#     ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
#     ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
#     ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']
# ]

dataset =  [
        ["apple", "banana", "cherry"],
        ["apple", "banana"],
        ["apple", "cherry"],
        ["banana", "cherry", "apple"],
        ["banana"]
    ]

# Initialize and fit TransactionEncoder
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)

# Convert the encoded array into a DataFrame
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the Apriori algorithm to find frequent item sets
frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

print("Frequent item sets:")
print(frequent_itemsets)

print("\nAssociation rules:")
print(rules)