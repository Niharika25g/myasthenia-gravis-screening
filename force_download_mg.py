import os
import urllib.request

# Use absolute path to ensure they go to the exact right folder
base_dir = os.path.abspath(os.path.dirname(__file__))
output_dir = os.path.join(base_dir, "dataset", "raw", "videos", "mg")
os.makedirs(output_dir, exist_ok=True)

videos = {
    "MG_Video_01.mp4": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8064859/bin/Video1.mp4",
    "MG_Video_02.mp4": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8064859/bin/Video2.mp4",
    "MG_Video_03.mp4": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8064859/bin/Video3.mp4",
}

print(f"[INFO] Downloading files directly into: {output_dir}")

for filename, url in videos.items():
    filepath = os.path.join(output_dir, filename)
    print(f"[DOWNLOADING] {filename}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
        with urllib.request.urlopen(req) as response, open(filepath, "wb") as out_file:
            out_file.write(response.read())
        print(f"[SUCCESS] Saved: {filename} ({os.path.getsize(filepath)} bytes)")
    except Exception as e:
        print(f"[ERROR] Failed to download {filename}: {e}")

print("[INFO] Download check complete! Refresh your File Explorer folder.")