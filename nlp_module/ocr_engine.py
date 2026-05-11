from paddleocr import PaddleOCR
import os

# Initialize OCR
ocr = PaddleOCR(lang='en')

# Image path
image_path = r"data/prescriptionimages/Code_Generated_image.png"

# Run OCR
result = ocr.ocr(image_path)

# Store extracted text
extracted_text = []

print("\nExtracted Text:\n")

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