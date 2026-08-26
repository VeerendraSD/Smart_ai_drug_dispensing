import re
import json
import uuid
import os
import sys

# =====================================
# PROJECT ROOT
# =====================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# =====================================
# OCR INPUT PATH
# =====================================

ocr_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "ocr_text.txt"
)

# Read OCR extracted text
with open(
    ocr_path,
    "r",
    encoding="utf-8"
) as file:

    text_data = file.read()

# Split lines
lines = text_data.splitlines()

# =====================================
# JSON STRUCTURE
# =====================================

prescription_data = {

    "prescription_id": "",

    "doctor": {
        "doctor_name": "",
        "department": ""
    },

    "patient": {
        "patient_id": "",
        "name": "",
        "age": 0,
        "gender": "",
        "existing_conditions": []
    },

    "medicines": [],

    "raw_text": text_data
}

# =====================================
# EXTRACT PRESCRIPTION ID
# =====================================

for line in lines:

    match = re.search(
        r'Prescription ID[:\s]*([A-Z0-9\-]+)',
        line,
        re.IGNORECASE
    )

    if match:

        prescription_data["prescription_id"] = (
            match.group(1)
        )

# =====================================
# EXTRACT PATIENT ID
# =====================================

for line in lines:

    match = re.search(
        r'^ID[:\s]*([A-Z0-9\-]+)',
        line,
        re.IGNORECASE
    )

    if match:

        prescription_data["patient"]["patient_id"] = (
            match.group(1)
        )

# =====================================
# EXTRACT PATIENT NAME
# =====================================

for line in lines:

    # Handles "Patient: Name", "Patient:Name" and "Patient Name" —
    # the colon-attached form is what the real OCR output produces
    # most of the time, and the old pattern (whitespace-only) missed it.
    match = re.search(
        r'Patient[:\s]+(.+)',
        line,
        re.IGNORECASE
    )

    if match:

        prescription_data["patient"]["name"] = (
            match.group(1).strip()
        )

# =====================================
# EXTRACT AGE AND GENDER
# =====================================

for line in lines:

    match = re.search(
        r'Age/Sex[:\s]*(\d+)\s*/\s*([MF])',
        line,
        re.IGNORECASE
    )

    if match:

        prescription_data["patient"]["age"] = int(
            match.group(1)
        )

        prescription_data["patient"]["gender"] = (
            match.group(2).upper()
        )

# =====================================
# EXTRACT DOCTOR
# =====================================

for line in lines:

    match = re.search(
        r'Dr\.\s*(.+?)\s*-\s*(.+)',
        line,
        re.IGNORECASE
    )

    if match:

        prescription_data["doctor"]["doctor_name"] = (
            match.group(1).strip()
        )

        prescription_data["doctor"]["department"] = (
            match.group(2).strip()
        )

# =====================================
# EXTRACT DIAGNOSIS / CONDITIONS
# =====================================

diagnosis_found = False

for line in lines:

    if "DIAGNOSIS" in line.upper():

        diagnosis_found = True
        continue

    if diagnosis_found:

        clean_line = line.strip()

        if clean_line == "":
            continue

        # The diagnosis section is expected to end at an "Rx" marker line,
        # but OCR frequently drops/garbles that single-word marker (e.g.
        # reads "Rx" as "x"). Relying on it alone let the medicine table,
        # signature line, and prescription ID leak into existing_conditions.
        # Also stop at any of those unambiguous section boundaries.
        is_rx_marker = clean_line.lower() == "rx"
        is_medicine_header = (
            "Medicine" in clean_line and "Dosage" in clean_line
        )
        is_medicine_row = "|" in clean_line
        is_signature_line = clean_line.lower().startswith("signed")
        is_prescription_id_line = re.search(
            r'Prescription ID[:\s]*[A-Z0-9\-]+',
            clean_line,
            re.IGNORECASE
        )

        if (
            is_rx_marker
            or is_medicine_header
            or is_medicine_row
            or is_signature_line
            or is_prescription_id_line
        ):
            break

        prescription_data[
            "patient"
        ][
            "existing_conditions"
        ].append(clean_line)

# =====================================
# EXTRACT MEDICINES
# =====================================

for line in lines:

    # Skip table header
    if (
        "Medicine" in line
        and "Dosage" in line
    ):
        continue

    # Detect medicine rows
    if "|" in line:

        parts = [
            p.strip()
            for p in line.split("|")
        ]

        if len(parts) >= 4:

            medicine_name = parts[0]

            # -----------------------------
            # DOSAGE
            # -----------------------------

            dosage_match = re.search(
                r'(\d+)',
                parts[1]
            )

            dosage = (
                int(dosage_match.group(1))
                if dosage_match
                else 0
            )

            # -----------------------------
            # FREQUENCY
            # -----------------------------

            frequency_match = re.search(
                r'(\d+)\s*x',
                parts[2],
                re.IGNORECASE
            )

            frequency = (
                int(frequency_match.group(1))
                if frequency_match
                else 0
            )

            # -----------------------------
            # ADD MEDICINE
            # -----------------------------

            prescription_data[
                "medicines"
            ].append({

                "medicine_id": str(
                    uuid.uuid4()
                ),

                "name": medicine_name,

                "dosage_mg": dosage,

                "frequency_per_day": frequency
            })

# =====================================
# VALIDATE EXTRACTED DATA
# =====================================

# A real prescription always lists at least one medicine. Zero medicines
# means OCR/parsing effectively failed, not that the prescription is
# genuinely empty. Continuing silently would let a garbage/empty
# prescription flow through preprocessing and the model as if it were a
# normal (and likely "safe") case, so fail loudly instead.
if not prescription_data["medicines"]:

    print(
        "\nERROR: No medicines could be extracted from the OCR text.",
        file=sys.stderr
    )

    print(
        "This usually means OCR failed to read the prescription image "
        "correctly, or the prescription format is unrecognized.",
        file=sys.stderr
    )

    sys.exit(1)

# =====================================
# CREATE CORRECT RAW JSON FOLDER
# =====================================

output_folder = os.path.join(
    PROJECT_ROOT,
    "data",
    "raw_json"
)

os.makedirs(
    output_folder,
    exist_ok=True
)

# =====================================
# CREATE OUTPUT FILENAME
# =====================================

if not prescription_data["prescription_id"]:

    prescription_data[
        "prescription_id"
    ] = str(uuid.uuid4())

output_file = os.path.join(
    output_folder,
    f"{prescription_data['prescription_id']}.json"
)

# -----------------------------
# Save JSON
# -----------------------------
with open(
    output_file,
    "w",
    encoding="utf-8"
) as json_file:

    json.dump(
        prescription_data,
        json_file,
        indent=4
    )


# -----------------------------
# SAVE EXACT JSON PATH
# -----------------------------

# Reuse the PROJECT_ROOT computed at the top of this file (relative to
# this script's own location) instead of a machine-specific absolute
# path — the previous hardcoded path broke the pipeline on any machine
# or checkout location other than the original developer's.
latest_json_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "latest_json_path.txt"
)

with open(
    latest_json_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        os.path.abspath(output_file)
    )


# -----------------------------
# Print Success Message
# -----------------------------

print("\nJSON file created successfully!")

print(
    f"\nSaved at: {os.path.abspath(output_file)}"
)

print(
    f"\nExact JSON path saved at: {latest_json_path}"
)