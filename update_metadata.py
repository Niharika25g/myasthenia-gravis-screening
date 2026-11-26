import os
import csv

METCSV_PATH = "dataset/metadata.csv"
AUDIO_CTRL_DIR = "dataset/raw/audio/control"
VIDEO_CTRL_DIR = "dataset/raw/videos/control"

# Read existing metadata if it exists
existing_rows = []
if os.path.exists(METCSV_PATH):
    with open(METCSV_PATH, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        existing_rows = list(reader)
else:
    existing_rows = [["filename", "subject_id", "modality", "label", "source", "split"]]

# Track existing filenames to avoid duplicates
existing_filenames = {row[0] for row in existing_rows[1:]}

new_rows = []

# Scan Control Audio Folder
if os.path.exists(AUDIO_CTRL_DIR):
    for filename in os.listdir(AUDIO_CTRL_DIR):
        if filename.endswith(".wav") and filename not in existing_filenames:
            subject_id = f"CTRL_SUB_{filename[:4]}"
            new_rows.append([filename, subject_id, "audio", 0, "RAVDESS_Control", "train"])

# Scan Control Video Folder
if os.path.exists(VIDEO_CTRL_DIR):
    for filename in os.listdir(VIDEO_CTRL_DIR):
        if filename.endswith(".mp4") and filename not in existing_filenames:
            subject_id = f"CTRL_SUB_{filename[:4]}"
            new_rows.append([filename, subject_id, "video", 0, "RAVDESS_Control", "train"])

if new_rows:
    existing_rows.extend(new_rows)
    with open(METCSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(existing_rows)
    print(f"[SUCCESS] Added {len(new_rows)} control files to {METCSV_PATH}")
else:
    print("[INFO] No new control files found to add (or they are already logged).")