from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import numpy as np

# Generate a larger classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train the MLP classifier
clf = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', max_iter=1000, random_state=42)

# Training loop with epoch printing
epochs_to_print = [100, 200, 300, 400, 500, 750, 1000]
for epoch in range(1, clf.max_iter + 1):
    clf.partial_fit(X_train, y_train, classes=np.unique(y_train))
    if epoch in epochs_to_print or epoch == clf.max_iter:
        y_pred_train = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        train_loss = clf.loss_
        print(f"Epoch {epoch}: Training Accuracy = {train_accuracy:.4f}, Training Loss = {train_loss:.4f}")

# Predict
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)