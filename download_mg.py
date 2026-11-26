import os
import urllib.request

# Target folder
output_dir = "dataset/raw/videos/mg"
os.makedirs(output_dir, exist_ok=True)

# Official URLs from PMC8064859
videos = {
    "MG_Video_01.mp4": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8064859/bin/Video1.mp4",
    "MG_Video_02.mp4": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8064859/bin/Video2.mp4",
    "MG_Video_03.mp4": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8064859/bin/Video3.mp4",
}

print("[INFO] Downloading real MG patient videos...")

for filename, url in videos.items():
  filepath = os.path.join(output_dir, filename)
  if os.path.exists(filepath):
    print(f"[SKIP] {filename} already exists.")
    continue

  print(f"[DOWNLOADING] {filename}...")
  try:
    # Add a user-agent header so PMC doesn't block the script
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response, open(
        filepath, "wb"
    ) as out_file:
      out_file.write(response.read())
    print(f"[SUCCESS] Downloaded: {filename}")
  except Exception as e:
    print(f"[ERROR] Failed to download {filename}: {e}")

print("[INFO] Download phase complete!")