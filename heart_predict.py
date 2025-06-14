# predict_heart.py

import sys
import numpy as np
import pandas as pd
import pickle

# Load the trained heart disease model and scaler
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("heart_scaler.pkl", "rb"))

# Use the correct column names from your dataset
columns = [
    "age", "sex", "cp", "trestbps", "chol", 
    "fbs", "restecg", "thalach", "exang", 
    "oldpeak", "slope", "ca", "thal"
]

# Read input from command-line arguments
input_data = np.array(sys.argv[1:], dtype=float).reshape(1, -1)

# Create DataFrame to match training format
df = pd.DataFrame(input_data, columns=columns)

# Scale the input
scaled = scaler.transform(df)

# Make prediction
result = model.predict(scaled)[0]

# Print the prediction result (0 = No disease, 1 = Disease)
print(result)
