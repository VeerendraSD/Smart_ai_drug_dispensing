import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OCR_PATH = ROOT / "data" / "processed" / "ocr_text.txt"
CURRENT_JSON = ROOT / "data" / "processed" / "current_prescription.json"
FEATURES_PATH = ROOT / "data" / "processed" / "feature_dataset.csv"


def run_pipeline(text):
    OCR_PATH.write_text(text, encoding="utf-8")
    subprocess.run(
        [sys.executable, str(ROOT / "nlp_module" / "json_generator.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    parsed = json.loads(CURRENT_JSON.read_text(encoding="utf-8"))
    if parsed["extraction_status"] != "ok":
        raise AssertionError(parsed["extraction_warnings"])
    subprocess.run(
        [sys.executable, str(ROOT / "datascience_module" / "preprocessing.py")],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    risk = json.loads(
        (ROOT / "data" / "processed" / "risk_analysis.json").read_text(
            encoding="utf-8"
        )
    )
    return parsed, FEATURES_PATH.read_text(encoding="utf-8"), risk


class PipelineRegressionTests(unittest.TestCase):
    def test_different_prescriptions_do_not_collapse_to_same_features(self):
        prescription_a, features_a, risk_a = run_pipeline(
            "Age/Sex: 30 / M\nParacetamol 500 mg 1x daily for 3 days\n"
        )
        prescription_b, features_b, risk_b = run_pipeline(
            "Age/Sex: 75 / F\nMorphine 20 mg 3x daily for 14 days\n"
        )

        self.assertNotEqual(prescription_a["medicines"], prescription_b["medicines"])
        self.assertNotEqual(features_a, features_b)
        self.assertNotEqual(risk_a["risk_score"], risk_b["risk_score"])
        self.assertNotEqual(
            prescription_a["patient"]["age"], prescription_b["patient"]["age"]
        )


if __name__ == "__main__":
    unittest.main()