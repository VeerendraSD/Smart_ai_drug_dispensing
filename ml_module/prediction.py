import pandas as pd
import joblib

# -----------------------------
# LOAD MODEL
# -----------------------------

model = joblib.load("ml_model/risk_classifier.pkl")

# -----------------------------
# LOAD NEW FEATURE DATA
# -----------------------------

df = pd.read_csv(
    "feature_dataset_1000_fixed.csv"
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
