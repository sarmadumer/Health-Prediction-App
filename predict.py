import sys
import numpy as np
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# Define feature names in the same order as training
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# Get values from command line and create DataFrame
input_data = np.array(sys.argv[1:], dtype=float).reshape(1, -1)
input_df = pd.DataFrame(input_data, columns=columns)

# Scale and predict
scaled = scaler.transform(input_df)
result = model.predict(scaled)[0]
confidence = model.predict_proba(scaled)[0][1]
print(result)

