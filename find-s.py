import pandas as pd

# Read training data from CSV file
data = pd.read_csv('training_data.csv')

print("Training Data:\n")
print(data)

# Attributes and target
concepts = data.iloc[:, :-1].values
target = data.iloc[:, -1].values

# Initialize hypothesis with first positive example
for i in range(len(target)):
    if target[i] == "Yes":
        hypothesis = concepts[i].copy()
        break

# Apply Find-S algorithm
for i in range(len(concepts)):
    if target[i] == "Yes":
        for j in range(len(hypothesis)):
            if hypothesis[j] != concepts[i][j]:
                hypothesis[j] = '?'

# Display final hypothesis
print("\nMost Specific Hypothesis:")
print(hypothesis)