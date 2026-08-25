import streamlit as st
import pandas as pd
import subprocess
import joblib
import os
import sys
import json

st.set_page_config(
    page_title="Smart AI Drug Dispensing",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Smart AI Drug Dispensing System")

PYTHON_PATH = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEBUG = os.getenv("SMART_DRUG_DEBUG", "0") == "1"


def project_path(*parts):
    return os.path.join(BASE_DIR, *parts)

# =====================================
# UPLOAD IMAGE
# =====================================

uploaded_file = st.file_uploader(
    "Upload Prescription Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    save_path = (
        project_path("data", "prescriptionimages", "Code_Generated_image.png")
    )

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ Prescription Uploaded")

    st.image(uploaded_file, width=500)

# =====================================
# ANALYZE BUTTON
# =====================================

if st.button("🔍 Analyze Prescription"):

    try:

        for stale_file in (
            project_path("data", "processed", "current_prescription.json"),
            project_path("data", "processed", "feature_dataset.csv"),
            project_path("data", "processed", "risk_analysis.json"),
        ):
            if os.path.exists(stale_file):
                os.remove(stale_file)

        with st.spinner("Analyzing Prescription..."):

            # OCR
            result = subprocess.run(
                [PYTHON_PATH, project_path("nlp_module", "ocr_engine.py")],
                cwd=BASE_DIR,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                st.error("OCR Engine Failed")
                st.code(result.stderr)
                st.stop()
            if DEBUG:
                st.code(result.stdout, language="text")

            # JSON Generation
            result = subprocess.run(
                [PYTHON_PATH, project_path("nlp_module", "json_generator.py")],
                cwd=BASE_DIR,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                st.error("JSON Generator Failed")
                st.code(result.stderr)
                st.stop()
            if DEBUG:
                st.code(result.stdout, language="text")

            # Feature Engineering
            result = subprocess.run(
                [PYTHON_PATH, project_path("datascience_module", "preprocessing.py")],
                cwd=BASE_DIR,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                st.error("Feature Engineering Failed")
                st.code(result.stderr)
                st.stop()
            if DEBUG:
                st.code(result.stdout, language="text")

        st.success("✅ Analysis Completed Successfully")

    except Exception as e:

        st.error(
            f"Pipeline Error: {e}"
        )

# =====================================
# OCR OUTPUT (OPTIONAL)
# =====================================

try:

    with open(
        "data/processed/ocr_text.txt",
        "r",
        encoding="utf-8"
    ) as f:

        ocr_text = f.read()

    with st.expander("📄 OCR Output"):

        st.text_area(
            "Extracted Text",
            ocr_text,
            height=250
        )

except FileNotFoundError:
    pass

# =====================================
# RISK ANALYSIS
# =====================================

st.header("🧠 Risk Analysis")

try:

    feature_df = pd.read_csv(
        project_path("data", "processed", "feature_dataset.csv")
    )
    if feature_df.empty:
        raise ValueError("Feature dataset is empty")

    row = feature_df.iloc[0]
    risk_path = project_path("data", "processed", "risk_analysis.json")
    with open(risk_path, "r", encoding="utf-8") as f:
        risk_analysis = json.load(f)

    st.write(feature_df)

    X = feature_df.drop(
        "requires_verification",
        axis=1,
        errors="ignore"
    )

    model_status = "not_loaded"
    prediction = None
    probability = None
    confidence = None
    model_path = project_path("ml_module", "risk_classifier.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        expected = list(getattr(model, "feature_names_in_", X.columns))
        if list(X.columns) != expected:
            raise ValueError(f"Feature mismatch. Expected {expected}, received {list(X.columns)}")
        prediction = model.predict(X)
        probability = model.predict_proba(X)
        confidence = round(float(max(probability[0])) * 100, 2)
        model_status = "loaded"
        if DEBUG:
            st.write("Model input:", X.to_dict(orient="records"))
            st.write("Model prediction:", prediction.tolist())
            st.write("Model probabilities:", probability.tolist())

    risk_score = risk_analysis.get("risk_score")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Age",
            int(row["age"])
        )

    with col2:
        st.metric(
            "Medicine Count",
            int(row["medicine_count"])
        )

    with col3:
        st.metric(
            "Toxicity Score",
            int(row["toxicity_total"])
        )

    st.divider()

    st.subheader("⚠ Risk Factors")

    risk_found = False

    if row["high_risk_present"] == 1:
        st.warning("High Risk Drug Present")
        risk_found = True

    if row["interaction_risk_present"] == 1:
        st.warning("Drug Interaction Risk Present")
        risk_found = True

    if row["black_box_present"] == 1:
        st.warning("Black Box Warning Present")
        risk_found = True

    if row["controlled_substance_present"] == 1:
        st.warning("Controlled Substance Present")
        risk_found = True

    if row["elderly"] == 1:
        st.warning("Elderly Patient")
        risk_found = True

    if row["high_dosage"] == 1:
        st.warning("High Dosage Prescription")
        risk_found = True

    if not risk_found:
        st.success("No major risk factors detected")

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model Confidence",
            f"{confidence}%" if confidence is not None else "Unavailable"
        )

    with col2:
        st.metric(
            "Rule Risk Score",
            f"{risk_score}/10" if risk_score is not None else "Unavailable"
        )

    st.caption(f"Rule analysis: {risk_analysis.get('prediction_status', 'unknown')}; ML model: {model_status}")
    if confidence is not None:
        st.caption(f"ML prediction confidence: {confidence}%")
    else:
        st.warning("ML confidence unavailable; manual review required.")

    st.divider()

    if risk_analysis.get("requires_verification") == 1 or model_status != "loaded":

        st.error(
            "🚨 VERIFICATION REQUIRED"
        )

        st.write(
            "Doctor/Hospital verification required before dispensing."
        )

    else:

        st.success(
            "✅ SAFE TO DISPENSE"
        )

        st.write(
            "Medicine can be dispensed safely."
        )

except (FileNotFoundError, ValueError, KeyError, OSError, ImportError):
    st.info(
        "Analysis is unavailable or incomplete. Review the OCR and extraction output before dispensing."
    )