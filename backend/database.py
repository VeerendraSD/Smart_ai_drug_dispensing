import sqlite3
import os
from datetime import datetime

# =====================================
# DATABASE PATH
# =====================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATABASE_FOLDER = os.path.join(
    PROJECT_ROOT,
    "data"
)

os.makedirs(
    DATABASE_FOLDER,
    exist_ok=True
)

DATABASE_PATH = os.path.join(
    DATABASE_FOLDER,
    "database.db"
)


# =====================================
# CREATE DATABASE TABLE
# =====================================

def initialize_database():

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prescription_records (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            prescription_id TEXT UNIQUE,

            patient_name TEXT,

            patient_age INTEGER,

            patient_gender TEXT,

            doctor_name TEXT,

            upload_date TEXT,

            upload_time TEXT
        )
    """)

    connection.commit()
    connection.close()


# =====================================
# SAVE PRESCRIPTION DETAILS
# =====================================

def save_prescription_details(prescription_data):

    patient = prescription_data.get(
        "patient",
        {}
    )

    doctor = prescription_data.get(
        "doctor",
        {}
    )

    prescription_id = prescription_data.get(
        "prescription_id"
    )

    patient_name = patient.get(
        "name",
        ""
    )

    patient_age = patient.get(
        "age"
    )

    patient_gender = patient.get(
        "gender",
        ""
    )

    doctor_name = doctor.get(
        "doctor_name",
        ""
    )

    # Current upload date/time
    now = datetime.now()

    upload_date = now.strftime(
        "%Y-%m-%d"
    )

    upload_time = now.strftime(
        "%H:%M:%S"
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    # Prevent duplicate prescription records
    cursor.execute("""
        INSERT OR IGNORE INTO prescription_records (
            prescription_id,
            patient_name,
            patient_age,
            patient_gender,
            doctor_name,
            upload_date,
            upload_time
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        prescription_id,
        patient_name,
        patient_age,
        patient_gender,
        doctor_name,
        upload_date,
        upload_time
    ))

    connection.commit()
    connection.close()


# =====================================
# GET PRESCRIPTION HISTORY
# =====================================

def get_prescription_history(limit=50):
    """Return the most recently analyzed prescriptions, newest first."""

    connection = sqlite3.connect(DATABASE_PATH)

    # Row factory lets each row be read out as a dict below instead of a
    # plain positional tuple, so the API response stays keyed by column
    # name even if the SELECT's column order ever changes.
    connection.row_factory = sqlite3.Row

    cursor = connection.cursor()

    cursor.execute("""
        SELECT
            prescription_id,
            patient_name,
            patient_age,
            patient_gender,
            doctor_name,
            upload_date,
            upload_time
        FROM prescription_records
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))

    records = [dict(row) for row in cursor.fetchall()]

    connection.close()

    return records


# =====================================
# INITIALIZE DATABASE
# =====================================

initialize_database()