import streamlit as st
import pandas as pd
import subprocess
import joblib
import os
import sys

st.set_page_config(
    page_title="Smart AI Drug Dispensing",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Smart AI Drug Dispensing System")

PYTHON_PATH = sys.executable

# =====================================
# UPLOAD IMAGE
# =====================================

uploaded_file = st.file_uploader(
    "Upload Prescription Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:

    save_path = (
        "data/prescriptionimages/"
        "Code_Generated_image.png"
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

        with st.spinner("Analyzing Prescription..."):

            # OCR
            result = subprocess.run(
                [PYTHON_PATH, "nlp_module/ocr_engine.py"],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                st.error("OCR Engine Failed")
                st.code(result.stderr)
                st.stop()

            # JSON Generation
            result = subprocess.run(
                [PYTHON_PATH, "nlp_module/json_generator.py"],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                st.error("JSON Generator Failed")
                st.code(result.stderr)
                st.stop()

            # Feature Engineering
            result = subprocess.run(
                [PYTHON_PATH, "datascience_module/preprocessing.py"],
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                st.error("Feature Engineering Failed")
                st.code(result.stderr)
                st.stop()

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

except:
    pass

# =====================================
# RISK ANALYSIS
# =====================================

st.header("🧠 Risk Analysis")

try:

    feature_df = pd.read_csv(
        "data/processed/feature_dataset.csv"
    )
    st.write(feature_df)
    model = joblib.load(
        "ml_module/risk_classifier.pkl"
    )

    X = feature_df.drop(
        "requires_verification",
        axis=1,
        errors="ignore"
    )

    prediction = model.predict(X)

    probability = model.predict_proba(X)

    confidence = round(
        max(probability[0]) * 100,
        2
    )

    risk_score = round(
        probability[0][1] * 100,
        2
    )

    row = feature_df.iloc[0]

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
            f"{confidence}%"
        )

    with col2:
        st.metric(
            "Risk Score",
            f"{risk_score}/100"
        )

    st.divider()

    if prediction[0] == 1:

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

except:
    st.info(
        "Upload and analyze a prescription to view results."
    )