import os
import json

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

ocr = PaddleOCR(
    lang="en",
    enable_mkldnn=False
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

else:

    print(
        "\nWARNING: OCR completed but no text was extracted."
    )


print(
    f"\nOCR text saved successfully at:\n{output_path}"
)