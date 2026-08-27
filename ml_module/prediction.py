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

# "unknown_medicine_present"/"unknown_medicine_names" describe medicines
# missing from the drug-risk database and, like "requires_verification",
# are not part of the model's trained feature set — they must be dropped
# before prediction, not fed to the model.
NON_FEATURE_COLUMNS = [
    "requires_verification",
    "unknown_medicine_present",
    "unknown_medicine_names",
]

X = df.drop(
    NON_FEATURE_COLUMNS,
    axis=1,
    errors="ignore"
)

# -----------------------------
# PREDICT
# -----------------------------

prediction = model.predict(X)

# -----------------------------
# UNKNOWN MEDICINE CHECK
# -----------------------------

# A medicine missing from the drug-risk database must never be treated as
# safe just because the model didn't see it — force verification instead
# of trusting the prediction.
unknown_medicine_present = bool(
    df.get("unknown_medicine_present", pd.Series([0])).iloc[0]
)

# -----------------------------
# DECISION
# -----------------------------

if unknown_medicine_present:

    unknown_names = df.get(
        "unknown_medicine_names", pd.Series([""])
    ).iloc[0]

    print("\n⚠ MEDICINE NOT FOUND")
    print(f"Unrecognized medicine(s): {unknown_names}")
    print("Manual verification required before dispensing.")

elif prediction[0] == 1:

    print("\n⚠ HIGH RISK PRESCRIPTION")
    print("Contact hospital for verification.")

else:
    print("\n SAFE PRESCRIPTION")
    print("Medicine can be dispensed.")
