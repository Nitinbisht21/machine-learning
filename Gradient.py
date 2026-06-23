import pandas as pd

# Read dataset
data = pd.read_csv("gradient.csv")

print("Training Data:\n")
print(data)

x = data["x"]
y = data["y"]

# Initialize parameters
m = 0
b = 0
L = 0.01
epochs = 1000
n = len(x)

# Gradient Descent
for i in range(epochs):

    y_pred = m * x + b

    Dm = (-2/n) * sum(x * (y - y_pred))
    Db = (-2/n) * sum(y - y_pred)

    m = m - L * Dm
    b = b - L * Db

print("\nSlope (m):")
print(m)

print("\nIntercept (b):")
print(b)

# Prediction
prediction = m * 6 + b

print("\nPrediction for x = 6:")
print(prediction)