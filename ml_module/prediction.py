import pandas as pd
import joblib
import os

# -----------------------------
# PROJECT PATH
# -----------------------------

# prediction.py is inside Smart_ai_drug_dispensing-main/ml_module/, so the
# project root is one level up. Computed from the script's own location
# instead of a hardcoded machine-specific path for portability.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# -----------------------------
# LOAD MODEL
# -----------------------------

model = joblib.load(
    os.path.join(PROJECT_ROOT, "ml_module", "risk_classifier.pkl")
)

# -----------------------------
# LOAD NEW FEATURE DATA
# -----------------------------

df = pd.read_csv(
    os.path.join(PROJECT_ROOT, "data", "processed", "feature_dataset.csv")
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
