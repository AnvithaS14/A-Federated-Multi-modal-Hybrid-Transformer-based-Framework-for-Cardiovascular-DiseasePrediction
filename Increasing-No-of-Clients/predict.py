import torch
import numpy as np
import joblib
from model import CVDModel

# Load model
input_dim = 13
model = CVDModel(input_dim)
model.load_state_dict(torch.load("cvd_model.pth"))
model.eval()

# Load scaler
scaler = joblib.load("scaler.pkl")

# Take user input
print("Enter patient details:")

age = float(input("Age: "))
sex = float(input("Sex (0=female, 1=male): "))
cp = float(input("Chest Pain Type (0-3): "))
trestbps = float(input("Resting BP: "))
chol = float(input("Cholesterol: "))
fbs = float(input("Fasting Blood Sugar (0/1): "))
restecg = float(input("Rest ECG (0-2): "))
thalach = float(input("Max Heart Rate: "))
exang = float(input("Exercise Angina (0/1): "))
oldpeak = float(input("Oldpeak: "))
slope = float(input("Slope (0-2): "))
ca = float(input("CA (0-3): "))
thal = float(input("Thal (1-3): "))

# Step 1: Create sample
sample = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                    thalach, exang, oldpeak, slope, ca, thal]])

# Step 2: Apply scaling
sample = scaler.transform(sample)

# Step 3: Convert to tensor
sample = torch.tensor(sample, dtype=torch.float32)

# Prediction
with torch.no_grad():
    prediction = model(sample)

print("\nRaw Prediction:", prediction.item())

# Threshold (adjust if needed)
if prediction.item() > 0.5:
    print("⚠️ High Risk of CVD")
else:
    print("✅ Low Risk of CVD")