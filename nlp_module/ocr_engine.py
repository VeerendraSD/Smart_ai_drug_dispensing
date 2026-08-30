import os
import json
import sys

# =====================================
# PROJECT ROOT
# =====================================

# ocr_engine.py is inside:
# Smart_ai_drug_dispensing-main/nlp_module/

# Therefore, go one level up to the main project folder
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# =====================================
# PADDLEOCR SETTINGS
# =====================================

os.environ["FLAGS_enable_pir_api"] = "0"

from paddleocr import PaddleOCR


# =====================================
# INITIALIZE OCR
# =====================================

# Model choice matters a lot here: the default PaddleOCR models
# (PP-OCRv6 "medium") take ~50s per image on CPU with no acceleration
# available (enable_mkldnn=False below is required — enabling it crashes
# outright on this paddlepaddle version). The "small" detection/recognition
# variants cut that to ~7s, verified to still parse prescription tables
# correctly. The doc-orientation/unwarping/textline-orientation
# sub-pipelines are also disabled since prescription photos are already
# upright, and skipping them removes unnecessary per-request model loads.
ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    text_detection_model_name="PP-OCRv6_small_det",
    text_recognition_model_name="PP-OCRv6_small_rec",
)


# =====================================
# IMAGE PATH
# =====================================

image_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "prescriptionimages",
    "Code_Generated_image.png"
)


# =====================================
# VERIFY IMAGE EXISTS
# =====================================

if not os.path.exists(image_path):

    print(
        f"\nERROR: Prescription image not found at:\n{image_path}",
        file=sys.stderr
    )

    sys.exit(1)


# =====================================
# RUN OCR
# =====================================

result = ocr.predict(image_path)


# =====================================
# EXTRACT TEXT
# =====================================

extracted_text = []

print("\nExtracted Text:\n")


for res in result:

    # Convert result to dictionary
    data = res.json

    if isinstance(data, str):
        data = json.loads(data)

    # PaddleOCR v3 stores results inside "res"
    if "res" in data:

        ocr_data = data["res"]

        if "rec_texts" in ocr_data:

            for text in ocr_data["rec_texts"]:

                extracted_text.append(text)

                print(text)


# =====================================
# PROCESSED FOLDER
# =====================================

processed_folder = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

os.makedirs(
    processed_folder,
    exist_ok=True
)


# =====================================
# SAVE OCR TEXT
# =====================================

output_path = os.path.join(
    processed_folder,
    "ocr_text.txt"
)


with open(
    output_path,
    "w",
    encoding="utf-8"
) as file:

    for text in extracted_text:
        file.write(text + "\n")


# =====================================
# FINAL STATUS
# =====================================

if extracted_text:

    print("\nOCR completed successfully.")

    print(
        f"Extracted {len(extracted_text)} lines of text."
    )

    print(
        f"\nOCR text saved successfully at:\n{output_path}"
    )

else:

    # No text extracted means the rest of the pipeline (JSON generation,
    # feature engineering, prediction) would silently run on an empty
    # prescription and could still produce a misleading "safe" result.
    # Fail the stage instead of letting app.py continue with old/empty data.
    print(
        "\nERROR: OCR completed but no text was extracted from the image.",
        file=sys.stderr
    )

    sys.exit(1)