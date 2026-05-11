import re
import json
import uuid
import os

# Read OCR extracted text from file
with open("data/processed/ocr_text.txt", "r") as file:
    text_data = file.read()

# Split lines
lines = text_data.split('\n')

# JSON structure
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

# -----------------------------
# Extract Prescription ID
# -----------------------------
for line in lines:
    match = re.search(r'Prescription ID[:\s]*([A-Z0-9\-]+)', line)

    if match:
        prescription_data["prescription_id"] = match.group(1)

# -----------------------------
# Extract Patient ID
# -----------------------------
for line in lines:
    match = re.search(r'ID[:\s]*([A-Z0-9\-]+)', line)

    if match:
        prescription_data["patient"]["patient_id"] = match.group(1)

# -----------------------------
# Extract Age and Gender
# -----------------------------
for line in lines:
    match = re.search(r'Age/Sex[:\s]*(\d+)\s*/\s*([MF])', line)

    if match:
        prescription_data["patient"]["age"] = int(match.group(1))
        prescription_data["patient"]["gender"] = match.group(2)

# -----------------------------
# Extract Diagnosis
# -----------------------------
diagnosis_found = False

for i, line in enumerate(lines):

    if "DIAGNOSIS" in line.upper():
        diagnosis_found = True
        continue

    if diagnosis_found:

        if line.strip() == "" or "Rx" in line:
            break

        prescription_data["patient"]["existing_conditions"].append(
            line.strip()
        )

# -----------------------------
# Extract Medicines
# -----------------------------
for line in lines:

    # Skip header line
    if "Medicine" in line and "Dosage" in line:
        continue

    # Detect medicine rows
    if "|" in line:

        parts = [p.strip() for p in line.split("|")]

        if len(parts) >= 4:

            medicine_name = parts[0]

            # Extract dosage number
            dosage_match = re.search(r'(\d+)', parts[1])

            dosage = int(dosage_match.group(1)) if dosage_match else 0

            # Extract frequency
            frequency_match = re.search(r'(\d+)x', parts[2])

            frequency = int(frequency_match.group(1)) if frequency_match else 0

            # Append medicine JSON
            prescription_data["medicines"].append({
                "medicine_id": str(uuid.uuid4()),
                "name": medicine_name,
                "dosage_mg": dosage,
                "frequency_per_day": frequency
            })

# -----------------------------
# Create raw_json folder
# -----------------------------
output_folder = "data/raw_json"

os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Create output filename
# -----------------------------
if prescription_data["prescription_id"] == "":
    prescription_data["prescription_id"] = str(uuid.uuid4())

output_file = os.path.join(
    output_folder,
    f"{prescription_data['prescription_id']}.json"
)

# -----------------------------
# Save JSON
# -----------------------------
with open(output_file, "w") as json_file:
    json.dump(prescription_data, json_file, indent=4)

# -----------------------------
# Print Success Message
# -----------------------------
print("\nJSON file created successfully!")
print(f"\nSaved at: {output_file}")