import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Sample data
data = {
    'age': [25, 30, 35, None, 40],
    'income': [50000, 60000, None, 70000, 80000],
    'gender': ['M', 'F', 'M', 'F', 'M'],
    'purchased': ['No', 'Yes', 'No', 'Yes', 'No']
}

# Step 1: Load Data into DataFrame
df = pd.DataFrame(data)

# Display the original DataFrame
print("Original DataFrame:")
print(df)

# Step 2: Handling Missing Values
# Fill missing values with mean for numeric columns
df.fillna(df.select_dtypes(include='number').mean(), inplace=True)

# Fill missing categorical values with the most frequent value
for column in df.select_dtypes(include='object').columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

# Display the DataFrame after handling missing values
print("\nDataFrame after handling missing values:")
print(df)

# Step 3: Handling Categorical Data
label_encoder = LabelEncoder()
df['gender'] = label_encoder.fit_transform(df['gender'])
df['purchased'] = label_encoder.fit_transform(df['purchased'])

# Display the DataFrame after encoding categorical data
print("\nDataFrame after encoding categorical data:")
print(df)

# Step 4: Feature Scaling
scaler = StandardScaler()
df[['age', 'income']] = scaler.fit_transform(df[['age', 'income']])

# Display the DataFrame after feature scaling
print("\nDataFrame after feature scaling:")
print(df)

# Step 5: Splitting Data
X = df.drop('purchased', axis=1)
y = df['purchased']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("\nShape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)

# Display the first few rows of the training and testing sets
print("\nFirst few rows of X_train:")
print(X_train.head())
print("\nFirst few rows of X_test:")
print(X_test.head())
