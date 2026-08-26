import streamlit as st
import pandas as pd
import subprocess
import joblib
import os
import sys
import hashlib
import json

from backend.database import save_prescription_details


st.set_page_config(
    page_title="Smart AI Drug Dispensing",
    page_icon="💊",
    layout="wide"
)

st.title("💊 Smart AI Drug Dispensing System")

PYTHON_PATH = sys.executable

# =====================================
# PROJECT ROOT
# =====================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)


def project_path(*parts):
    return os.path.join(
        PROJECT_ROOT,
        *parts
    )


# =====================================
# PATHS
# =====================================

IMAGE_PATH = project_path(
    "data",
    "prescriptionimages",
    "Code_Generated_image.png"
)

OCR_TEXT_PATH = project_path(
    "data",
    "processed",
    "ocr_text.txt"
)

FEATURE_CSV_PATH = project_path(
    "data",
    "processed",
    "feature_dataset.csv"
)

MODEL_PATH = project_path(
    "ml_module",
    "risk_classifier.pkl"
)

JSON_FOLDER = project_path(
    "data",
    "raw_json"
)


# =====================================
# SESSION STATE
# =====================================

if "analyzed_hash" not in st.session_state:
    st.session_state.analyzed_hash = None

if "analysis" not in st.session_state:
    st.session_state.analysis = None

if "pipeline_logs" not in st.session_state:
    st.session_state.pipeline_logs = None


# =====================================
# UPLOAD IMAGE
# =====================================

uploaded_file = st.file_uploader(
    "Upload Prescription Image",
    type=["png", "jpg", "jpeg"]
)

current_hash = None


if uploaded_file:

    file_bytes = uploaded_file.getvalue()

    current_hash = hashlib.sha256(
        file_bytes
    ).hexdigest()

    os.makedirs(
        os.path.dirname(IMAGE_PATH),
        exist_ok=True
    )

    with open(
        IMAGE_PATH,
        "wb"
    ) as f:
        f.write(file_bytes)

    st.success(
        "✅ Prescription Uploaded"
    )

    st.image(
        uploaded_file,
        width=500
    )

    if (
        st.session_state.analyzed_hash
        != current_hash
    ):

        st.info(
            "This image hasn't been analyzed yet. "
            "Click 'Analyze Prescription' below to "
            "run the pipeline on it."
        )


# =====================================
# ANALYZE BUTTON
# =====================================

if st.button(
    "🔍 Analyze Prescription"
):

    if not uploaded_file:

        st.warning(
            "Please upload a prescription image first."
        )

    else:

        stages = [

            (
                "OCR Engine",
                project_path(
                    "nlp_module",
                    "ocr_engine.py"
                )
            ),

            (
                "JSON Generator",
                project_path(
                    "nlp_module",
                    "json_generator.py"
                )
            ),

            (
                "Feature Engineering",
                project_path(
                    "datascience_module",
                    "preprocessing.py"
                )
            ),
        ]

        logs = []

        pipeline_ok = True


        with st.spinner(
            "Analyzing Prescription..."
        ):

            for (
                stage_name,
                script_path
            ) in stages:

                result = subprocess.run(
                    [
                        PYTHON_PATH,
                        script_path
                    ],
                    capture_output=True,
                    text=True,
                    cwd=PROJECT_ROOT
                )

                logs.append(
                    f"----- {stage_name} -----\n"
                    f"[stdout]\n"
                    f"{result.stdout}\n"
                    f"[stderr]\n"
                    f"{result.stderr}\n"
                )

                if result.returncode != 0:

                    st.error(
                        f"{stage_name} Failed"
                    )

                    st.code(
                        result.stderr
                        or result.stdout
                        or "No output."
                    )

                    pipeline_ok = False

                    break


        st.session_state.pipeline_logs = (
            "\n".join(logs)
        )


        # =====================================
        # LOAD RESULTS AFTER PIPELINE
        # =====================================

        if pipeline_ok:

            try:

                # =====================================
                # FIND GENERATED JSON
                # =====================================

                if not os.path.exists(
                    JSON_FOLDER
                ):

                    raise FileNotFoundError(
                        "JSON folder does not exist."
                    )


                json_files = [

                    os.path.join(
                        JSON_FOLDER,
                        filename
                    )

                    for filename in os.listdir(
                        JSON_FOLDER
                    )

                    if filename.lower().endswith(
                        ".json"
                    )
                ]


                if not json_files:

                    raise FileNotFoundError(
                        "No JSON file was generated."
                    )


                # Use the JSON generated most recently
                latest_json = max(
                    json_files,
                    key=os.path.getmtime
                )


                # =====================================
                # LOAD JSON
                # =====================================

                with open(
                    latest_json,
                    "r",
                    encoding="utf-8"
                ) as f:

                    prescription_data = json.load(
                        f
                    )


                # =====================================
                # SAVE PATIENT DETAILS TO SQLITE
                # =====================================

                save_prescription_details(
                    prescription_data
                )


                # =====================================
                # LOAD OCR OUTPUT
                # =====================================

                with open(
                    OCR_TEXT_PATH,
                    "r",
                    encoding="utf-8"
                ) as f:

                    ocr_text = f.read()


                # =====================================
                # LOAD FEATURE DATA
                # =====================================

                feature_df = pd.read_csv(
                    FEATURE_CSV_PATH
                )


                if feature_df.empty:

                    raise ValueError(
                        "Feature dataset is empty — "
                        "preprocessing produced no rows."
                    )


                # =====================================
                # LOAD MODEL
                # =====================================

                model = joblib.load(
                    MODEL_PATH
                )


                # =====================================
                # PREPARE MODEL INPUT
                # =====================================

                X = feature_df.drop(
                    "requires_verification",
                    axis=1,
                    errors="ignore"
                )


                # =====================================
                # MODEL PREDICTION
                # =====================================

                prediction = model.predict(
                    X
                )

                probability = model.predict_proba(
                    X
                )


                # =====================================
                # STORE ANALYSIS
                # =====================================

                st.session_state.analysis = {

                    "ocr_text":
                        ocr_text,

                    "feature_df":
                        feature_df,

                    "prediction":
                        prediction,

                    "probability":
                        probability,

                    "prescription_data":
                        prescription_data
                }


                st.session_state.analyzed_hash = (
                    current_hash
                )


                st.success(
                    "✅ Analysis Completed Successfully"
                )


            except Exception as e:

                st.session_state.analysis = None

                st.session_state.analyzed_hash = None

                st.error(
                    f"Pipeline Error while loading "
                    f"results: {e}"
                )


# =====================================
# PIPELINE LOGS
# =====================================

with st.expander(
    "🛠 Pipeline Logs (debug)"
):

    st.code(
        st.session_state.pipeline_logs
        or "No pipeline run yet."
    )


# =====================================
# RESULTS READY CHECK
# =====================================

results_ready = (

    uploaded_file is not None

    and st.session_state.analysis is not None

    and st.session_state.analyzed_hash
        == current_hash
)


if not uploaded_file:

    st.info(
        "Upload and analyze a prescription "
        "to view results."
    )

elif not results_ready:

    st.info(
        "Results shown below always belong "
        "to the currently uploaded image. "
        "Click 'Analyze Prescription' to "
        "generate them."
    )


# =====================================
# DISPLAY RESULTS
# =====================================

if results_ready:

    analysis = (
        st.session_state.analysis
    )

    ocr_text = analysis[
        "ocr_text"
    ]

    feature_df = analysis[
        "feature_df"
    ]

    prediction = analysis[
        "prediction"
    ]

    probability = analysis[
        "probability"
    ]

    prescription_data = analysis[
        "prescription_data"
    ]


    # =====================================
    # PATIENT / DOCTOR DETAILS
    # =====================================

    patient = prescription_data.get(
        "patient",
        {}
    )

    doctor = prescription_data.get(
        "doctor",
        {}
    )


    st.header(
        "👤 Prescription Details"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Patient Name",
            patient.get(
                "name",
                "Unknown"
            )
        )


    with col2:

        st.metric(
            "Age",
            patient.get(
                "age",
                "Unknown"
            )
        )


    with col3:

        st.metric(
            "Gender",
            patient.get(
                "gender",
                "Unknown"
            )
        )


    st.write(
        f"**Doctor:** "
        f"{doctor.get('doctor_name', 'Unknown')}"
    )


    st.divider()


    # =====================================
    # OCR OUTPUT
    # =====================================

    with st.expander(
        "📄 OCR Output"
    ):

        st.text_area(
            "Extracted Text",
            ocr_text,
            height=250
        )


    # =====================================
    # RISK ANALYSIS
    # =====================================

    st.header(
        "🧠 Risk Analysis"
    )


    confidence = round(
        max(
            probability[0]
        ) * 100,
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
            int(
                row[
                    "medicine_count"
                ]
            )
        )


    with col3:

        st.metric(
            "Toxicity Score",
            int(
                row[
                    "toxicity_total"
                ]
            )
        )


    st.divider()


    # =====================================
    # RISK FACTORS
    # =====================================

    st.subheader(
        "⚠ Risk Factors"
    )


    risk_found = False


    if row[
        "high_risk_present"
    ] == 1:

        st.warning(
            "High Risk Drug Present"
        )

        risk_found = True


    if row[
        "interaction_risk_present"
    ] == 1:

        st.warning(
            "Drug Interaction Risk Present"
        )

        risk_found = True


    if row[
        "black_box_present"
    ] == 1:

        st.warning(
            "Black Box Warning Present"
        )

        risk_found = True


    if row[
        "controlled_substance_present"
    ] == 1:

        st.warning(
            "Controlled Substance Present"
        )

        risk_found = True


    if row[
        "elderly"
    ] == 1:

        st.warning(
            "Elderly Patient"
        )

        risk_found = True


    if row[
        "high_dosage"
    ] == 1:

        st.warning(
            "High Dosage Prescription"
        )

        risk_found = True


    if not risk_found:

        st.success(
            "No major risk factors detected"
        )


    st.divider()


    # =====================================
    # MODEL METRICS
    # =====================================

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


    # =====================================
    # FINAL DECISION
    # =====================================

    if prediction[0] == 1:

        st.error(
            "🚨 VERIFICATION REQUIRED"
        )

        st.write(
            "Doctor/Hospital verification "
            "required before dispensing."
        )


    else:

        st.success(
            "✅ SAFE TO DISPENSE"
        )

        st.write(
            "Medicine can be dispensed safely."
        )


    st.divider()


    # =====================================
    # RAW FEATURE DATA
    # =====================================

    with st.expander(
        "🔬 Raw Feature Data"
    ):

        st.dataframe(
            feature_df
        )