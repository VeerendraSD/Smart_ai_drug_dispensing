# Smart AI Drug Dispensing System

## Overview

The Smart AI Drug Dispensing System is an AI-powered healthcare project designed to automate prescription analysis, medical risk detection, and verification support before medicine dispensing. The system uses OCR, NLP, feature engineering, and healthcare risk analysis to process printed prescriptions and generate ML-ready healthcare datasets.

---

## Problem Statement

Traditional prescription verification is manual, time-consuming, and prone to medication errors. Existing systems mainly focus on prescription digitization or medicine dispensing but lack intelligent medical risk analysis and automated verification workflows.

This project aims to reduce medication errors, improve patient safety, and support healthcare professionals using AI-assisted prescription analysis.

---

## Features

* Prescription Image Processing
* OCR-based Text Extraction using PaddleOCR
* NLP-based Structured JSON Generation
* Medical Feature Engineering
* Drug Risk Analysis
* Dosage Safety Evaluation
* Drug Interaction Detection
* Automated Verification Labeling
* ML-Ready Dataset Generation
* AI-Based Risk Prediction (trained Random Forest classifier)
* Web Interface for Prescription Upload & Analysis (FastAPI backend + browser frontend)
* Prescription Record History (SQLite)

## Technologies Used

* Python
* PaddleOCR
* NLP & Regex
* pandas
* scikit-learn (Random Forest)
* FastAPI + Uvicorn
* SQLite

## Getting Started

1. Install [Python 3.12](https://www.python.org/downloads/) (required — `paddlepaddle`/`paddleocr` don't yet support newer Python versions).
2. Double-click `start_app.bat` (Windows). On first run it creates a local `.venv`, installs everything in `requirements.txt`, then starts the server and opens `http://127.0.0.1:8000` in your browser. Later runs skip straight to starting the server.
3. Upload a prescription image on the page to run it through the full pipeline (OCR → structuring → feature engineering → risk prediction).

To run manually instead of via the launcher:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

## Current Status

*Phase 1 & 2 Completed*

* OCR Pipeline
* JSON Structuring
* Feature Engineering
* Risk Label Generation
* ML-Ready Dataset Creation
* Risk Prediction Model Trained & Deployed
* FastAPI Backend + Web UI
* Prescription Record History (SQLite)

---

## Future Scope

* Automated Voice Verification Calls
* Flutter Frontend (mobile)
* Cloud Deployment
* Expanded Database Integration (cloud-hosted patient history, multi-user access)
* Authentication & Access Control

---

## Team Roles

* NLP Module – OCR & Text Structuring
* Data Science Module – Feature Engineering & Dataset Generation
* ML Module – Risk Prediction Model
* Cloud/Backend Module – APIs & Database Integration

---

## Objective

To develop an AI-powered healthcare system capable of intelligent prescription analysis, medical risk identification, and automated verification support before medicine dispensing.
