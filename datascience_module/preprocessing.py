import json
import pandas as pd

# -----------------------------
# LOAD PRESCRIPTION JSON
# -----------------------------

with open("../data/raw_json/RX-2024001.json") as f:
    data = json.load(f)

# -----------------------------
# LOAD DRUG RISK DATABASE CSV
# -----------------------------

drug_db = pd.read_csv("../data/drug_risk_database.csv")

# -----------------------------
# BASIC FEATURES
# -----------------------------

age = data["patient"]["age"]

medicine_count = len(data["medicines"])

total_dosage = sum(
    med["dosage_mg"]
    for med in data["medicines"]
)

has_conditions = 1 if data["patient"]["existing_conditions"] else 0

elderly = 1 if age >= 60 else 0

high_dosage = 1 if total_dosage > 1000 else 0

# -----------------------------
# ADVANCED DRUG RISK FEATURES
# -----------------------------

toxicity_total = 0

high_risk_present = 0

interaction_risk_present = 0

black_box_present = 0

controlled_substance_present = 0

dose_ratio_total = 0

max_dose_ratio = 0

# -----------------------------
# MATCH MEDICINES WITH CSV
# -----------------------------

for med in data["medicines"]:

    med_name = med["name"]

    matched = drug_db[
        drug_db["medicine_name"]
        .str.lower() == med_name.lower()
    ]

    if not matched.empty:

        row = matched.iloc[0]

        # -----------------------------
        # TOXICITY FEATURES
        # -----------------------------

        toxicity_total += int(row["toxicity_score"])

        if row["high_risk_drug"] == 1:
            high_risk_present = 1

        if row["interaction_risk_flag"] == 1:
            interaction_risk_present = 1

        if row["black_box_warning"] == 1:
            black_box_present = 1

        if row["controlled_substance"] == 1:
            controlled_substance_present = 1

        # -----------------------------
        # DOSE RATIO FEATURE
        # -----------------------------

        max_safe_dose = row["max_safe_daily_dose_mg"]

        prescribed_dose = med["dosage_mg"]

        if max_safe_dose > 0:

            dose_ratio = prescribed_dose / max_safe_dose

            dose_ratio_total += dose_ratio

            if dose_ratio > max_dose_ratio:
                max_dose_ratio = dose_ratio

# -----------------------------
# RISK SCORING SYSTEM
# -----------------------------

risk_score = 0

if high_risk_present == 1:
    risk_score += 3

if controlled_substance_present == 1:
    risk_score += 3

if interaction_risk_present == 1:
    risk_score += 2

if black_box_present == 1:
    risk_score += 2

if elderly == 1:
    risk_score += 1

if high_dosage == 1:
    risk_score += 3

if toxicity_total > 10:
    risk_score += 2

if medicine_count > 5:
    risk_score += 1

if max_dose_ratio > 0.8:
    risk_score += 3

# -----------------------------
# TARGET LABEL
# -----------------------------

if risk_score >= 6:
    requires_verification = 1
else:
    requires_verification = 0

# -----------------------------
# FINAL FEATURE DICTIONARY
# -----------------------------

features = {
    "age": age,
    "medicine_count": medicine_count,
    "total_dosage": total_dosage,
    "has_conditions": has_conditions,
    "elderly": elderly,
    "high_dosage": high_dosage,

    "toxicity_total": toxicity_total,
    "high_risk_present": high_risk_present,
    "interaction_risk_present": interaction_risk_present,
    "black_box_present": black_box_present,
    "controlled_substance_present":
        controlled_substance_present,

    "dose_ratio_total": round(dose_ratio_total, 3),
    "max_dose_ratio": round(max_dose_ratio, 3),

    # TARGET LABEL
    "requires_verification":
        requires_verification
}

# -----------------------------
# PRINT FEATURES
# -----------------------------

print("\nExtracted Features:\n")

print(features)

print("\nGenerated Risk Score:\n")

print(risk_score)

# -----------------------------
# CONVERT TO DATAFRAME
# -----------------------------

df = pd.DataFrame([features])

print("\nDataFrame:\n")

print(df)

# -----------------------------
# SAVE PROCESSED CSV
# -----------------------------

df.to_csv(
    "../data/processed/feature_dataset.csv",
    index=False
)

print(
    "\nProcessed dataset saved successfully!"
)