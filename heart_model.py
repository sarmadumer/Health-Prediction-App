# train_heart_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

# Step 1: Load Dataset
df = pd.read_csv("heart.csv")

# Step 2: Rename target column for clarity (optional)
df.rename(columns={"target": "HeartDisease"}, inplace=True)

# Step 3: Separate features and target
X = df.drop("HeartDisease", axis=1)
y = df["HeartDisease"]

# Step 4: Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 6: Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Step 7: Save Model and Scaler
with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("heart_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Heart disease model and scaler saved successfully.")
