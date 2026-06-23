import pandas as pd
from sklearn.linear_model import LinearRegression

# Read dataset
data = pd.read_csv("linear.csv")

print("Training Data:\n")
print(data)

# Features and target
x = data[["Hours"]]
y = data["Marks"]

# Train model
model = LinearRegression()
model.fit(x, y)

# Predict marks for 9 hours of study
result = model.predict(pd.DataFrame([[9]], columns=["Hours"]))

print("\nPredicted Marks:")
print(result)