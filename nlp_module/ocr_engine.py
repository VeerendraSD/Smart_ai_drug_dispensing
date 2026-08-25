from paddleocr import PaddleOCR
import os

# Initialize OCR
ocr = PaddleOCR(lang='en')

# Image path
image_path = r"data/prescriptionimages/Code_Generated_image.png"

# Run OCR
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Prescription image not found: {image_path}")

result = ocr.ocr(image_path)

# Store extracted text
extracted_text = []

print("\nExtracted Text:\n")

if not result or not result[0]:
    raise ValueError("OCR returned no text")

for line in result[0]:

    text = line[1][0]

    extracted_text.append(text)

    print(text)

# -----------------------------
# Create processed folder
# -----------------------------
os.makedirs("data/processed", exist_ok=True)

# -----------------------------
# Save OCR text into file
# -----------------------------
output_path = "data/processed/ocr_text.txt"

with open(output_path, "w", encoding="utf-8") as file:

    for text in extracted_text:
        file.write(text + "\n")

print(f"\nOCR text saved successfully at:\n{output_path}")