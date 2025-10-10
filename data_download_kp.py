import os
import json
import requests
import mimetypes
from tqdm import tqdm
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

# ==============================
# CONFIGURATION
# ==============================
SAVE_DIR = "laion_images_kp"
METADATA_FILE = "laion_metadata_kp.jsonl"
N_SAMPLES = 1_000_000       # total samples you want
MAX_WORKERS = 32            # number of parallel download threads
TIMEOUT = 5                 # timeout per image in seconds

# ==============================
# SETUP
# ==============================
os.makedirs(SAVE_DIR, exist_ok=True)
dataset = load_dataset("laion/relaion2B-en-research-safe", streaming=True, split="train")
subset = dataset.shuffle(seed=42).take(N_SAMPLES)
subset = subset.remove_columns([
    'similarity', 'hash', 'pwatermark', 'punsafe', 'key',
    'status', 'error_message', 'width', 'height', 'original_width',
    'original_height', 'exif', 'md5'
])

# ==============================
# DOWNLOAD FUNCTION
# ==============================
def get_extension_from_response(url, response):
    """Infer file extension from content-type header or URL."""
    content_type = response.headers.get('Content-Type', '').lower()
    if 'image/' in content_type:
        ext = content_type.split('/')[-1]
        # Normalize common MIME anomalies
        if ext in ['jpeg', 'pjpeg']:
            ext = 'jpg'
        elif ext == 'svg+xml':
            ext = 'svg'
        return f".{ext}"
    else:
        # fallback from URL
        ext = os.path.splitext(url.split("?")[0])[1]
        if ext and len(ext) <= 5:
            return ext
    return ".jpg"  # default fallback

def download_image(idx, url):
    """Download image from URL and save locally (any image type)."""
    try:
        response = requests.get(url, timeout=TIMEOUT)
        if response.status_code == 200 and 'image' in response.headers.get('Content-Type', ''):
            ext = get_extension_from_response(url, response)
            path = os.path.join(SAVE_DIR, f"{idx}{ext}")
            with open(path, "wb") as f:
                f.write(response.content)
            return True
    except Exception:
        pass
    return False

# ==============================
# MAIN PIPELINE
# ==============================
print("starting download")
success_count = 0
with open(METADATA_FILE, "w", encoding="utf-8") as meta_f, \
     ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

    futures = {}
    for idx, sample in enumerate(tqdm(islice(subset, N_SAMPLES), total=N_SAMPLES, desc="Submitting jobs")):
        url = sample.get('url')
        text = sample.get('caption', "")
        if not url:
            continue

        # Submit async download
        futures[executor.submit(download_image, idx, url)] = (idx, url, text)

        # Write metadata immediately
        json.dump({"id": idx, "url": url, "text": text}, meta_f)
        meta_f.write("\n")

    # Collect results
    for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading images"):
        idx, url, text = futures[future]
        if future.result():
            success_count += 1

print(f"✅ Completed. Downloaded {success_count} images successfully to '{SAVE_DIR}'")
print(f"Metadata saved in '{METADATA_FILE}'")
