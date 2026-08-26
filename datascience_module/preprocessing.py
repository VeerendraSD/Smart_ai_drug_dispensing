import json
import pandas as pd
import os
import sys


# =====================================
# PROJECT PATH
# =====================================

# preprocessing.py is inside Smart_ai_drug_dispensing-main/datascience_module/,
# so the project root is one level up. Computing this from the script's own
# location (instead of a hardcoded machine-specific path) makes the pipeline
# work from any checkout location or machine.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# =====================================
# PATHS
# =====================================

LATEST_JSON_PATH_FILE = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "latest_json_path.txt"
)

DRUG_DATABASE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "drug_risk_database.csv"
)

OUTPUT_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "feature_dataset.csv"
)


# =====================================
# READ EXACT JSON PATH
# =====================================

if not os.path.exists(LATEST_JSON_PATH_FILE):

    raise FileNotFoundError(
        "latest_json_path.txt not found. "
        "Run json_generator.py first."
    )


with open(
    LATEST_JSON_PATH_FILE,
    "r",
    encoding="utf-8"
) as file:

    json_path = file.read().strip()


# =====================================
# CHECK JSON EXISTS
# =====================================

if not os.path.exists(json_path):

    raise FileNotFoundError(
        f"JSON file not found:\n{json_path}"
    )


print("\n=====================================")
print("Using JSON file:")
print(json_path)
print("=====================================\n")


# =====================================
# LOAD JSON
# =====================================

with open(
    json_path,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


# =====================================
# LOAD DRUG DATABASE
# =====================================

drug_db = pd.read_csv(
    DRUG_DATABASE_PATH
)


# =====================================
# BASIC FEATURES
# =====================================

age = data["patient"]["age"]

medicine_count = len(
    data["medicines"]
)

total_dosage = sum(
    med["dosage_mg"]
    for med in data["medicines"]
)

has_conditions = (
    1
    if data["patient"]["existing_conditions"]
    else 0
)

elderly = (
    1
    if age >= 60
    else 0
)

high_dosage = (
    1
    if total_dosage > 1000
    else 0
)


# =====================================
# ADVANCED RISK FEATURES
# =====================================

toxicity_total = 0

high_risk_present = 0

interaction_risk_present = 0

black_box_present = 0

controlled_substance_present = 0

dose_ratio_total = 0

max_dose_ratio = 0


# =====================================
# MATCH MEDICINES
# =====================================

for med in data["medicines"]:

    med_name = med["name"]

    matched = drug_db[
        drug_db["medicine_name"]
        .str.lower()
        ==
        med_name.lower()
    ]

    if matched.empty:

        # An unmatched medicine contributes no toxicity/risk flags at all,
        # which can silently understate risk. This isn't fatal (the drug
        # database may just be incomplete), but it must be visible rather
        # than a silent no-op.
        print(
            f"WARNING: '{med_name}' not found in drug_risk_database.csv "
            "— it will not contribute to risk scoring.",
            file=sys.stderr
        )

    else:

        row = matched.iloc[0]


        # -----------------------------
        # TOXICITY
        # -----------------------------

        toxicity_total += int(
            row["toxicity_score"]
        )


        # -----------------------------
        # RISK FLAGS
        # -----------------------------

        if row["high_risk_drug"] == 1:

            high_risk_present = 1


        if row["interaction_risk_flag"] == 1:

            interaction_risk_present = 1


        if row["black_box_warning"] == 1:

            black_box_present = 1


        if row["controlled_substance"] == 1:

            controlled_substance_present = 1


        # -----------------------------
        # DOSE RATIO
        # -----------------------------

        max_safe_dose = row[
            "max_safe_daily_dose_mg"
        ]

        prescribed_dose = med[
            "dosage_mg"
        ]

        if max_safe_dose > 0:

            dose_ratio = (
                prescribed_dose
                /
                max_safe_dose
            )

            dose_ratio_total += dose_ratio

            if dose_ratio > max_dose_ratio:

                max_dose_ratio = dose_ratio


# =====================================
# RISK SCORE
# =====================================

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


# =====================================
# TARGET
# =====================================

if risk_score >= 4:

    requires_verification = 1

else:

    requires_verification = 0


# =====================================
# FEATURE DICTIONARY
# =====================================

features = {

    "age":
        age,

    "medicine_count":
        medicine_count,

    "total_dosage":
        total_dosage,

    "has_conditions":
        has_conditions,

    "elderly":
        elderly,

    "high_dosage":
        high_dosage,

    "toxicity_total":
        toxicity_total,

    "high_risk_present":
        high_risk_present,

    "interaction_risk_present":
        interaction_risk_present,

    "black_box_present":
        black_box_present,

    "controlled_substance_present":
        controlled_substance_present,

    "dose_ratio_total":
        round(
            dose_ratio_total,
            3
        ),

    "max_dose_ratio":
        round(
            max_dose_ratio,
            3
        ),
}


# =====================================
# PRINT
# =====================================

print("\nExtracted Features:\n")

print(features)

print("\nGenerated Risk Score:")

print(risk_score)


# =====================================
# DATAFRAME
# =====================================

df = pd.DataFrame(
    [features]
)


print("\nDataFrame:\n")

print(df)


# =====================================
# SAVE FEATURES
# =====================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "\nProcessed features saved successfully!"
)

print(
    f"\nSaved at:\n{OUTPUT_PATH}"
)