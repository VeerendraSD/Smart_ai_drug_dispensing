import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# -----------------------------
# LOAD MODEL
# -----------------------------

model = joblib.load(ROOT / "ml_module" / "risk_classifier.pkl")

# -----------------------------
# LOAD NEW FEATURE DATA
# -----------------------------

df = pd.read_csv(
    ROOT / "data" / "processed" / "feature_dataset.csv"
)

X = df.drop(
    "requires_verification",
    axis=1,
    errors="ignore"
)

expected = list(getattr(model, "feature_names_in_", X.columns))
if list(X.columns) != expected:
    raise ValueError(
        f"Feature mismatch. Expected {expected}, received {list(X.columns)}"
    )

# -----------------------------
# PREDICT
# -----------------------------

prediction = model.predict(X)
probabilities = model.predict_proba(X)

print("FEATURES:", X.to_dict(orient="records"))
print("FEATURE SHAPE:", X.shape)
print("MODEL PREDICTION:", prediction.tolist())
print("MODEL PROBABILITIES:", probabilities.tolist())

# -----------------------------
# DECISION
# -----------------------------

if prediction[0] == 1:

    print("\n⚠ HIGH RISK PRESCRIPTION")
    print("Contact hospital for verification.")

else:
    print("\n SAFE PRESCRIPTION")
    print("Medicine can be dispensed.")
