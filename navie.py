import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load dataset
data = pd.read_csv('spam.csv')

print("Dataset:\n")
print(data)

# Features and target
X = data['Message']
y = data['Category']

# Convert text into numerical vectors
cv = CountVectorizer()
X = cv.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nPredicted Output:\n")
print(y_pred)

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nPrecision:")
print(precision_score(y_test, y_pred, pos_label='Spam'))

print("\nRecall:")
print(recall_score(y_test, y_pred, pos_label='Spam'))