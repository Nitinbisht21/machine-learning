import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# Load Wine dataset
wine = load_wine()
# Features and target
X = wine.data
y = wine.target
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)
# Create SVM model
model = SVC(kernel='linear')
# Train model
model.fit(X_train, y_train)
# Prediction
y_pred = model.predict(X_test)
print("Predicted Output:")
print(y_pred)
# Accuracy
print("Accuracy:")
print(accuracy_score(y_test, y_pred))
# Precision
print("Precision:")
print(precision_score(y_test, y_pred, average='macro'))
# Recall
print("Recall:")
print(recall_score(y_test, y_pred, average='macro'))

# F1 Score
print("F1 Score:")
print(f1_score(y_test, y_pred, average='macro'))