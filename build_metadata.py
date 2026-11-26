import os
import csv

def generate_metadata():
    metadata_path = "dataset/metadata.csv"
    os.makedirs("dataset", exist_ok=True)
    
    rows = []
    # Header definition
    header = ["filename", "subject_id", "modality", "label", "source", "split"]
    
    # Define directories to scan
    # Label 1 = MG, Label 0 = Control
    folders_to_scan = [
        {"path": "dataset/raw/videos/mg", "modality": "video", "label": "1", "source": "PMC8064859"},
        {"path": "dataset/raw/audio/mg", "modality": "audio", "label": "1", "source": "PMC8064859"},
        {"path": "dataset/raw/videos/control", "modality": "video", "label": "0", "source": "Control_Dataset"},
        {"path": "dataset/raw/audio/control", "modality": "audio", "label": "0", "source": "Control_Dataset"}
    ]
    
    counter = 1
    for info in folders_to_scan:
        folder = info["path"]
        if os.path.exists(folder):
            files = os.listdir(folder)
            for filename in files:
                if filename.lower().endswith(('.mp4', '.avi', '.mov', '.wav', '.mp3', '.m4a')):
                    subject_id = f"SUB_{counter:03d}"
                    split = "train" if counter % 5 != 0 else "test" # automatically split some into test
                    rows.append([filename, subject_id, info["modality"], info["label"], info["source"], split])
                    counter += 1

    # Write to metadata.csv
    with open(metadata_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
        
    print(f"[SUCCESS] Generated {metadata_path} with {len(rows)} total records across your folders!")

if __name__ == "__main__":
    generate_metadata()