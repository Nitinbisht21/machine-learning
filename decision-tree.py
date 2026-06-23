import pandas as pd 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.preprocessing import LabelEncoder  
data = pd.read_csv("tennis.csv") 
print("Training Data:\n") 
print(data) 
le = LabelEncoder() 
Outlook = le.fit_transform(data["Outlook"]) 
Temperature = le.fit_transform(data["Temperature"]) 
Humidity = le.fit_transform(data["Humidity"]) 
Wind = le.fit_transform(data["Wind"]) 
x = list(zip(Outlook, Temperature, Humidity, Wind)) 
y = le.fit_transform(data["Play Tennis"]) 
model = DecisionTreeClassifier() 
model.fit(x, y) 
sample = [[2, 2, 0, 1]] 
prediction = model.predict(sample) 
print("\nPrediction for new sample:") 
if prediction == 1: 
    print("Yes") 
else: 
    print("No")