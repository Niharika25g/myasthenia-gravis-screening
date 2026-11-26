import os
import csv
from moviepy import VideoFileClip

# --- Configuration ---
CATEGORIES = {"mg": 1, "control": 0}
METADATA_PATH = os.path.join("dataset", "metadata.csv")

# Ensure metadata file has a header if it doesn't exist
if not os.path.exists(METADATA_PATH):
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "subject_id", "modality", "label", "source", "split"])

# Load existing filenames to avoid duplicates in CSV
existing_filenames = set()
with open(METADATA_PATH, mode="r", encoding="utf-8") as f:
    reader = csv.reader(f)
    next(reader, None) # Skip header
    for row in reader:
        if row:
            existing_filenames.add(row[0])

def append_to_csv(filename, subject_id, modality, label, source="real_dataset", split="train"):
    if filename in existing_filenames:
        return
    with open(METADATA_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([filename, subject_id, modality, label, source, split])
    existing_filenames.add(filename)
    print(f"[CSV] Added entry for {filename} ({modality}, label={label})")

# --- Process Videos and Audio ---
for cat, label_val in CATEGORIES.items():
    video_dir = os.path.join("dataset", "raw", "videos", cat)
    audio_dir = os.path.join("dataset", "raw", "audio", cat)
    os.makedirs(audio_dir, exist_ok=True)

    if not os.path.exists(video_dir):
        continue

    for filename in os.listdir(video_dir):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(video_dir, filename)
            base_name = os.path.splitext(filename)[0]
            audio_filename = f"{base_name}.wav"
            audio_path = os.path.join(audio_dir, audio_filename)
            subject_id = f"SUB_{base_name.upper()}"

            # 1. Automatically register video in metadata.csv
            append_to_csv(filename, subject_id, "video", label_val)

            # 2. Extract audio if it doesn't exist yet
            if not os.path.exists(audio_path):
                try:
                    print(f"[EXTRACTING] {filename} -> {audio_filename}...")
                    video_clip = VideoFileClip(video_path)
                    if video_clip.audio is not None:
                        video_clip.audio.write_audiofile(audio_path, fps=16000, logger=None)
                        video_clip.close()
                        print(f"[SUCCESS] Saved audio to {audio_dir}")
                    else:
                        print(f"[WARNING] Video {filename} has no audio track.")
                        video_clip.close()
                except Exception as e:
                    print(f"[ERROR] Failed extracting audio from {filename}: {e}")
            else:
                print(f"[SKIP] Audio file already exists for {filename}")

            # 3. Automatically register extracted audio in metadata.csv
            append_to_csv(audio_filename, subject_id, "audio", label_val)

print("[INFO] Processing and automatic CSV row generation completed successfully!")