import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL
# -----------------------------

model = joblib.load("C:/Users/veere/OneDrive/Desktop/Smart_ai_drug_dispensing-main/ml_module/risk_classifier.pkl")

# -----------------------------
# LOAD NEW FEATURE DATA
# -----------------------------

df = pd.read_csv(
    "C:/Users/veere/OneDrive/Desktop/Smart_ai_drug_dispensing-main/data/processed/feature_dataset.csv"
)

X = df.drop(
    "requires_verification",
    axis=1,
    errors="ignore"
)

# -----------------------------
# PREDICT
# -----------------------------

prediction = model.predict(X)

# -----------------------------
# DECISION
# -----------------------------

if prediction[0] == 1:

    print("\n⚠ HIGH RISK PRESCRIPTION")
    print("Contact hospital for verification.")

else:
    print("\n SAFE PRESCRIPTION")
    print("Medicine can be dispensed.")
