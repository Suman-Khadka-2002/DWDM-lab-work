import matplotlib.pyplot as plt
import pandas as pd

data = {
    "feature1": [1, 2, 3, 4, 5, 6, 8, 9, 10, None],
    "feature2": ["A", "A", "B", "B", "A", "C", "B", "A", "C", None],
    "target": [100, 90, 90, 110, 110, 105, 125, 140, 115, 120],
}

data = pd.DataFrame(data)

# Data cleaning

# Handle missing values (replace with mean for simplicity)
data["feature1"] = data["feature1"].fillna(data["feature1"].mean())
data["target"] = data["target"].fillna(data["target"].mean())

# Convert columns to NumPy arrays
feature1 = data["feature1"].to_numpy()
feature2 = data["feature2"].to_numpy()
target = data["target"].to_numpy()

plt.hist(target, rwidth=0.7)
plt.title("Lab 1")
plt.ylabel("count")
plt.xlabel("target")
plt.show()

# Print output (example)
print(data.head())  # Display the first few rows of the cleaned data