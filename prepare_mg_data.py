import os
import csv
from moviepy import VideoFileClip  # Fixed import for modern MoviePy versions

# Define paths
VIDEO_MG_DIR = "dataset/raw/videos/mg"
AUDIO_MG_DIR = "dataset/raw/audio/mg"
METCSV_PATH = "dataset/metadata.csv"

os.makedirs(AUDIO_MG_DIR, exist_ok=True)
os.makedirs("dataset", exist_ok=True)

# Initialize metadata CSV header
metadata_rows = [["filename", "subject_id", "modality", "label", "source", "split"]]

print("[INFO] Starting MG video-to-audio extraction pipeline...")

# Process MG Videos
if os.path.exists(VIDEO_MG_DIR):
    for filename in os.listdir(VIDEO_MG_DIR):
        if filename.endswith(".mp4"):
            video_path = os.path.join(VIDEO_MG_DIR, filename)
            base_name = os.path.splitext(filename)[0]
            wav_filename = f"{base_name}_audio.wav"
            wav_path = os.path.join(AUDIO_MG_DIR, wav_filename)
            
            # 1. Extract Audio using MoviePy
            try:
                clip = VideoFileClip(video_path)
                if clip.audio is not None:
                    clip.audio.write_audiofile(wav_path, fps=16000, nbytes=2, codec='pcm_s16le', logger=None)
                    print(f"[SUCCESS] Extracted audio: {wav_filename}")
                else:
                    print(f"[WARNING] No audio track found in {filename}")
                clip.close()
            except Exception as e:
                print(f"[ERROR] Failed to process {filename}: {e}")
                continue
            
            # 2. Add records to metadata schema
            subject_id = f"MG_SUB_{base_name[-2:]}"
            metadata_rows.append([filename, subject_id, "video", 1, "PMC8064859", "train"])
            metadata_rows.append([wav_filename, subject_id, "audio", 1, "PMC8064859", "train"])

# Write master metadata.csv
with open(METCSV_PATH, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(metadata_rows)

print(f"[INFO] Pipeline complete! Metadata successfully logged to {METCSV_PATH}")