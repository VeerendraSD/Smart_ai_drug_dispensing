import json
import pandas as pd
import os
import re
import sys
import difflib
import itertools


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

DRUG_INTERACTIONS_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "drug_interactions.csv"
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
# NAME NORMALIZATION (for OCR-tolerant matching)
# =====================================

def normalize_medicine_name(name):
    """Normalize a medicine name for matching so minor OCR artifacts
    (extra/irregular spacing, inconsistent capitalization, stray
    punctuation) don't cause a real match to be missed."""

    name = str(name).strip().lower()

    # Collapse any run of whitespace to a single space.
    name = re.sub(r"\s+", " ", name)

    # Drop characters that aren't letters, digits or spaces (OCR noise
    # like stray periods/commas/pipes that leak in from the source image).
    name = re.sub(r"[^a-z0-9 ]", "", name)

    return name.strip()


drug_db["_normalized_name"] = drug_db["medicine_name"].apply(
    normalize_medicine_name
)

_KNOWN_NORMALIZED_NAMES = drug_db["_normalized_name"].tolist()


# =====================================
# DRUG-DRUG INTERACTION LOOKUP
# =====================================

# data/drug_interactions.csv is a small, curated set of well-established,
# textbook-level drug-drug interactions (~30 pairs) covering some of the
# medicines in drug_risk_database.csv. It is NOT an exhaustive or
# clinically-validated interaction database (e.g. Lexicomp/Micromedex/First
# Databank) — it exists to demonstrate real pairwise interaction checking
# between the SPECIFIC medicines on one prescription, which is different
# from (and complements) drug_risk_database.csv's "interaction_risk_flag"
# column, which only marks that a single drug is *involved in* some known
# interaction, without checking what it was actually co-prescribed with.
_interactions_df = pd.read_csv(DRUG_INTERACTIONS_PATH)

# Keyed by a frozenset of the two normalized drug names so lookup doesn't
# care which order the medicines appear in on the prescription.
_INTERACTION_LOOKUP = {}
for _, _row in _interactions_df.iterrows():
    _key = frozenset({
        normalize_medicine_name(_row["drug_a"]),
        normalize_medicine_name(_row["drug_b"]),
    })
    _INTERACTION_LOOKUP[_key] = {
        "severity": str(_row["severity"]).strip().lower(),
        "description": str(_row["description"]).strip(),
    }


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

# Medicines that could not be matched to any row in drug_risk_database.csv,
# even after OCR-tolerant normalization/fuzzy matching. These must NEVER be
# treated as "safe" or contribute a default/zero risk value silently — they
# are flagged below and force manual verification regardless of the
# computed risk score.
unmatched_medicines = []

# Canonical (drug_risk_database.csv) names of medicines that WERE matched,
# used below to check every pair against the interaction lookup. An
# unmatched medicine's real identity isn't known, so it can't be checked
# for interactions — it's already flagged separately as unknown.
matched_medicine_names = []


# =====================================
# MATCH MEDICINES
# =====================================

for med in data["medicines"]:

    med_name = med["name"]

    normalized_name = normalize_medicine_name(med_name)

    matched = drug_db[
        drug_db["_normalized_name"] == normalized_name
    ]

    if matched.empty:

        # Exact normalized match failed — allow a close fuzzy match to
        # absorb minor OCR misreads (e.g. a dropped/substituted letter),
        # but keep the cutoff strict so genuinely different drug names
        # are never silently merged.
        close_matches = difflib.get_close_matches(
            normalized_name,
            _KNOWN_NORMALIZED_NAMES,
            n=1,
            cutoff=0.85
        )

        if close_matches:
            matched = drug_db[
                drug_db["_normalized_name"] == close_matches[0]
            ]

    if matched.empty:

        # An unmatched medicine must not silently contribute zero
        # toxicity/risk flags as if it were known-safe — the drug database
        # may simply be incomplete, or OCR may have mangled the name
        # beyond recognition. Either way this must be visible and force
        # manual verification, never a silent no-op.
        print(
            f"WARNING: '{med_name}' not found in drug_risk_database.csv "
            "— flagging as unknown medicine requiring manual verification.",
            file=sys.stderr
        )

        unmatched_medicines.append(med_name)

    else:

        row = matched.iloc[0]

        matched_medicine_names.append(row["medicine_name"])


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
# CHECK PAIRWISE DRUG-DRUG INTERACTIONS
# =====================================

# Unlike interaction_risk_present above (a static per-drug flag from
# drug_risk_database.csv), this checks every actual PAIR of medicines on
# THIS prescription against the curated interaction list — e.g. it only
# fires for Warfarin when it's genuinely co-prescribed with Ibuprofen, not
# just because Warfarin appears on some interaction list at all.
detected_interactions = []

for drug_a, drug_b in itertools.combinations(matched_medicine_names, 2):

    key = frozenset({
        normalize_medicine_name(drug_a),
        normalize_medicine_name(drug_b),
    })

    interaction = _INTERACTION_LOOKUP.get(key)

    if interaction:
        detected_interactions.append({
            "drug_a": drug_a,
            "drug_b": drug_b,
            "severity": interaction["severity"],
            "description": interaction["description"],
        })

drug_interaction_pairs_present = 1 if detected_interactions else 0

major_interaction_present = any(
    d["severity"] == "major" for d in detected_interactions
)


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


for _interaction in detected_interactions:
    risk_score += 3 if _interaction["severity"] == "major" else 2


# =====================================
# UNKNOWN MEDICINE FLAG
# =====================================

unknown_medicine_present = (
    1
    if unmatched_medicines
    else 0
)


# =====================================
# TARGET
# =====================================

# A medicine that couldn't be matched to the drug-risk database must never
# be treated as safe just because it added nothing to risk_score — force
# manual verification whenever any medicine is unknown, regardless of the
# score computed from the medicines that WERE matched. A major drug-drug
# interaction between two medicines that WERE matched must force it too,
# for the same reason.
if risk_score >= 4 or unknown_medicine_present == 1 or major_interaction_present:

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

    # NOTE: like unknown_medicine_present/unknown_medicine_names below,
    # these two columns are NOT part of the model's trained feature set —
    # callers must drop them before calling model.predict(). They describe
    # actual pairwise interactions found between the medicines on THIS
    # prescription (see detected_interactions above), which is more
    # specific than interaction_risk_present's per-drug flag.
    "drug_interaction_pairs_present":
        drug_interaction_pairs_present,

    "drug_interaction_pairs_details":
        "; ".join(
            f"{d['drug_a']} + {d['drug_b']} ({d['severity']}): {d['description']}"
            for d in detected_interactions
        ),

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

    # NOTE: these two columns describe medicines the drug-risk database has
    # no entry for. They are NOT part of the model's trained feature set —
    # callers (e.g. the backend API) must drop them before calling
    # model.predict(), exactly like "requires_verification" — and are only
    # here so downstream code can detect an unknown medicine and force
    # manual verification instead of trusting the model's prediction.
    "unknown_medicine_present":
        unknown_medicine_present,

    "unknown_medicine_names":
        "; ".join(unmatched_medicines),
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