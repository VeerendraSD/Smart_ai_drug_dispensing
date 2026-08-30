"""
FastAPI backend for the Smart AI Drug Dispensing System.

Replaces the previous Streamlit UI. This module exposes:
  - GET  /                → the frontend page (backend/static/index.html)
  - POST /api/analyze     → runs the full pipeline on an uploaded
                             prescription image and returns the results
                             as JSON for the frontend to render.

The underlying pipeline stages (OCR, JSON generation, feature
engineering) are unchanged — they are invoked exactly as they were from
the Streamlit app, as separate scripts via subprocess, using whichever
Python interpreter is running this server (sys.executable). This keeps
the same reliability guarantees the pipeline scripts already have
(project-root-relative paths, hard failure on bad OCR/parsing) without
duplicating that logic here.
"""

import os
import sys
import json
import subprocess
from datetime import datetime

import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .database import save_prescription_details, get_prescription_history

# =====================================
# PATHS
# =====================================

# backend/app.py -> project root is one level up
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

PYTHON_PATH = sys.executable


def project_path(*parts):
    return os.path.join(PROJECT_ROOT, *parts)


IMAGE_PATH = project_path("data", "prescriptionimages", "Code_Generated_image.png")
OCR_TEXT_PATH = project_path("data", "processed", "ocr_text.txt")
FEATURE_CSV_PATH = project_path("data", "processed", "feature_dataset.csv")
MODEL_PATH = project_path("ml_module", "risk_classifier.pkl")
LATEST_JSON_PATH_FILE = project_path("data", "processed", "latest_json_path.txt")
PREDICTIONS_DIR = project_path("data", "predictions")

PIPELINE_STAGES = [
    ("OCR Engine", project_path("nlp_module", "ocr_engine.py")),
    ("JSON Generator", project_path("nlp_module", "json_generator.py")),
    ("Feature Engineering", project_path("datascience_module", "preprocessing.py")),
]

app = FastAPI(title="Smart AI Drug Dispensing System")


# =====================================
# STATIC FRONTEND
# =====================================

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


# =====================================
# HISTORY ENDPOINT
# =====================================

@app.get("/api/history")
def get_history(limit: int = 50):
    """Return the most recently analyzed prescriptions, newest first, from
    the local SQLite record store — the same records save_prescription_details()
    writes at the end of every /api/analyze call."""

    return get_prescription_history(limit=limit)


# =====================================
# PIPELINE STAGE RUNNER
# =====================================

def run_stage(stage_name, script_path):
    """Run one pipeline stage. Raises HTTPException with the stage's own
    stderr/stdout on failure so the frontend can show exactly what broke,
    instead of continuing on to the next stage with stale/missing data."""

    result = subprocess.run(
        [PYTHON_PATH, script_path],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "No output."
        raise HTTPException(
            status_code=422,
            detail=f"{stage_name} failed:\n{detail}",
        )

    return result.stdout


# =====================================
# ANALYZE ENDPOINT
# =====================================

@app.post("/api/analyze")
async def analyze_prescription(file: UploadFile = File(...)):

    if file.content_type not in ("image/png", "image/jpeg", "image/jpg"):
        raise HTTPException(
            status_code=400,
            detail="Only PNG or JPEG prescription images are supported.",
        )

    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Save the CURRENT uploaded image to the fixed path every stage script
    # expects. This always overwrites whatever was analyzed before, so a
    # new upload can never be analyzed against an old image.
    os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)
    with open(IMAGE_PATH, "wb") as f:
        f.write(file_bytes)

    # ---- run pipeline stages in order, stop on first failure ----
    for stage_name, script_path in PIPELINE_STAGES:
        run_stage(stage_name, script_path)

    # ---- load the JSON produced by THIS run ----
    # Read the exact path json_generator.py just wrote, rather than
    # scanning data/raw_json for the newest file by mtime — this is the
    # one source of truth for "the JSON from the current run".
    if not os.path.exists(LATEST_JSON_PATH_FILE):
        raise HTTPException(
            status_code=500,
            detail="Pipeline did not produce latest_json_path.txt.",
        )

    with open(LATEST_JSON_PATH_FILE, "r", encoding="utf-8") as f:
        json_path = f.read().strip()

    if not os.path.exists(json_path):
        raise HTTPException(
            status_code=500,
            detail=f"Generated JSON file not found: {json_path}",
        )

    with open(json_path, "r", encoding="utf-8") as f:
        prescription_data = json.load(f)

    # ---- persist patient/prescription summary ----
    try:
        save_prescription_details(prescription_data)
    except Exception as e:
        # Recording history must not block returning the analysis itself.
        print(f"WARNING: failed to save prescription record: {e}", file=sys.stderr)

    # ---- load OCR text ----
    with open(OCR_TEXT_PATH, "r", encoding="utf-8") as f:
        ocr_text = f.read()

    # ---- load features generated by THIS run ----
    feature_df = pd.read_csv(FEATURE_CSV_PATH)

    if feature_df.empty:
        raise HTTPException(
            status_code=500,
            detail="Feature dataset is empty — preprocessing produced no rows.",
        )

    row = feature_df.iloc[0]

    # ---- unknown medicine check ----
    # "unknown_medicine_present"/"unknown_medicine_names" describe medicines
    # preprocessing couldn't match in drug_risk_database.csv (even after
    # OCR-tolerant normalization). They are NOT part of the model's trained
    # feature set, so they must be dropped before prediction — but they
    # still have to override the final verdict below, since an unmatched
    # medicine must never be silently treated as safe just because it
    # contributed no risk features.
    unknown_medicine_present = bool(row.get("unknown_medicine_present", 0))

    # An empty "unknown_medicine_names" cell round-trips through the CSV as
    # a real empty string, but pandas reads that back as NaN (a float), not
    # "". `str(nan)` is the literal text "nan", and since NaN is truthy in
    # Python, `nan or ""` doesn't catch it either — so without this check
    # the list below would end up as ["nan"] instead of [] whenever there
    # are no unknown medicines. Normalize NaN to "" first to avoid that.
    raw_unknown_names = row.get("unknown_medicine_names", "")
    if pd.isna(raw_unknown_names):
        raw_unknown_names = ""

    unknown_medicine_names = [
        name.strip()
        for name in str(raw_unknown_names).split(";")
        if name.strip()
    ]

    # ---- run the model ----
    model = joblib.load(MODEL_PATH)
    NON_FEATURE_COLUMNS = [
        "requires_verification",
        "unknown_medicine_present",
        "unknown_medicine_names",
    ]
    X = feature_df.drop(NON_FEATURE_COLUMNS, axis=1, errors="ignore")
    prediction = model.predict(X)
    probability = model.predict_proba(X)

    confidence = round(float(max(probability[0])) * 100, 2)
    risk_score = round(float(probability[0][1]) * 100, 2)

    risk_factors = []
    if unknown_medicine_present:
        risk_factors.append(
            "Medicine Not Found: " + ", ".join(unknown_medicine_names)
        )
    if row["high_risk_present"] == 1:
        risk_factors.append("High Risk Drug Present")
    if row["interaction_risk_present"] == 1:
        risk_factors.append("Drug Interaction Risk Present")
    if row["black_box_present"] == 1:
        risk_factors.append("Black Box Warning Present")
    if row["controlled_substance_present"] == 1:
        risk_factors.append("Controlled Substance Present")
    if row["elderly"] == 1:
        risk_factors.append("Elderly Patient")
    if row["high_dosage"] == 1:
        risk_factors.append("High Dosage Prescription")

    # An unmatched medicine must always require manual verification,
    # regardless of what the model predicted from the medicines it did
    # recognize — never let it fall through as "safe".
    requires_verification = bool(prediction[0] == 1) or unknown_medicine_present

    patient = prescription_data.get("patient", {})
    doctor = prescription_data.get("doctor", {})

    response_data = {
        "patient": {
            "name": patient.get("name") or "Unknown",
            "age": patient.get("age"),
            "gender": patient.get("gender") or "Unknown",
        },
        "doctor": {
            "name": doctor.get("doctor_name") or "Unknown",
            "department": doctor.get("department") or "Unknown",
        },
        "ocr_text": ocr_text,
        "metrics": {
            "age": int(row["age"]),
            "medicine_count": int(row["medicine_count"]),
            "toxicity_total": int(row["toxicity_total"]),
        },
        "risk_factors": risk_factors,
        "confidence": confidence,
        "risk_score": risk_score,
        "requires_verification": requires_verification,
        "unknown_medicine_present": unknown_medicine_present,
        "unknown_medicine_names": unknown_medicine_names,
        "feature_data": json.loads(feature_df.to_json(orient="records"))[0],
    }

    # ---- persist a per-analysis prediction record ----
    # Kept separate from the SQLite history: this is the FULL analysis
    # result (risk factors, confidence, feature breakdown) for whoever
    # ran this specific prescription through the pipeline, filed under
    # its prescription_id — not just the patient/doctor summary row that
    # save_prescription_details() writes to the database. Timestamped so
    # re-analyzing the same prescription_id doesn't overwrite an earlier
    # run's record.
    try:
        os.makedirs(PREDICTIONS_DIR, exist_ok=True)
        prescription_id = prescription_data.get("prescription_id") or "unknown"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prediction_file = os.path.join(
            PREDICTIONS_DIR, f"{prescription_id}_{timestamp}.json"
        )
        with open(prediction_file, "w", encoding="utf-8") as f:
            json.dump(response_data, f, indent=2)
    except Exception as e:
        # Saving the prediction record must not block returning the
        # analysis itself.
        print(f"WARNING: failed to save prediction record: {e}", file=sys.stderr)

    return response_data
