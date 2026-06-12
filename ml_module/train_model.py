import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import joblib

# -----------------------------
# LOAD DATASET
# -----------------------------

df = pd.read_csv(
    "feature_dataset_1000_fixed.csv"
)

print("\nDataset Shape:")
print(df.shape)

# -----------------------------
# FEATURES AND TARGET
# -----------------------------

X = df.drop(
    "requires_verification",
    axis=1
)

y = df["requires_verification"]

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# RANDOM FOREST MODEL
# -----------------------------

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)

# -----------------------------
# TRAIN MODEL
# -----------------------------

model.fit(X_train, y_train)

# -----------------------------
# PREDICTIONS
# -----------------------------

y_pred = model.predict(X_test)

# -----------------------------
# EVALUATION
# -----------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred
    )
)

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance:\n")
print(importance)

# -----------------------------
# SAVE MODEL
# -----------------------------

joblib.dump(
    model,
    "ml_model/risk_classifier.pkl"
)

print(
    "\nModel saved successfully!"
)
