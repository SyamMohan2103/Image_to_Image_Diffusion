import os
import glob
import time
import math
import random
import requests
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import pandas as pd
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==============================
# Config
# ==============================
OUT_DIR = "laion_images"
BATCH_SIZE = 2000                 # how many per batch
MAX_SAMPLES = 10000000                # total samples desired
EXCEL_PREFIX = "laion_subset"     # Excel files: laion_subset_batchN.xlsx
WORKERS = 32                      # parallel downloads per batch
REQ_TIMEOUT = 6                   # seconds per request
MAX_RETRIES = 2                   # per-URL retry attempts (in addition to pool retries)
BACKOFF_BASE = 0.5                # backoff base seconds

os.makedirs(OUT_DIR, exist_ok=True)

# ==============================
# Resume logic: check existing batches
# ==============================
existing_batches = sorted(glob.glob(f"{EXCEL_PREFIX}_batch*.xlsx"))
if existing_batches:
    last_batch_file = existing_batches[-1]
    last_batch_num = int(last_batch_file.split("batch")[-1].split(".")[0])
    print(f"➡️ Resuming from batch {last_batch_num+1} (found {last_batch_file})")
    start_batch = last_batch_num + 1
    start_index = (last_batch_num) * BATCH_SIZE
else:
    print("➡️ Starting fresh")
    start_batch = 1
    start_index = 0

# ==============================
# Load dataset (streaming mode)
# ==============================
ds = load_dataset("laion/relaion2B-en-research-safe", split="train", streaming=True)
subset = ds.shuffle(seed=42).take(MAX_SAMPLES)

subset = subset.remove_columns([
    'similarity', 'hash', 'pwatermark', 'punsafe', 'key',
    'status', 'error_message', 'width', 'height', 'original_width',
    'original_height', 'exif', 'md5'
])

# ==============================
# HTTP session with pooling & retries
# ==============================
def make_session():
    sess = requests.Session()
    retry = Retry(
        total=3,                # connection-level retries
        read=3,
        connect=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=WORKERS, pool_maxsize=WORKERS)
    sess.mount("http://", adapter)
    sess.mount("https://", adapter)
    # Lightweight default headers
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; laion-downloader/1.0)"
    })
    return sess

session = make_session()

# ==============================
# Image download worker
# ==============================
def fetch_and_save(item):
    idx, sample, batch_num = item
    url = sample.get("url")
    caption = sample.get("caption")
    if not url:
        return idx, None, "no_url"

    # Retry loop beyond requests built-in retry for PIL/open failures
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQ_TIMEOUT, stream=True)
            if resp.status_code != 200 or resp.content is None:
                raise RuntimeError(f"bad_status:{resp.status_code}")

            # Limit content read size to avoid huge downloads (e.g., 30 MB)
            content = resp.content
            if len(content) > 30 * 1024 * 1024:
                raise RuntimeError("too_large")

            # Validate and convert
            with Image.open(BytesIO(content)) as im:
                im = im.convert("RGB")
                img_path = os.path.join(OUT_DIR, f"batch{batch_num}_{idx}.jpg")
                im.save(img_path, format="JPEG", quality=90, optimize=True)
                return idx, {"image_path": img_path, "caption": caption}, None
        except (requests.RequestException, UnidentifiedImageError, OSError, RuntimeError) as e:
            if attempt < MAX_RETRIES:
                sleep_s = BACKOFF_BASE * (2 ** attempt) + random.random() * 0.2
                time.sleep(sleep_s)
            else:
                return idx, None, str(e)
    return idx, None, "unknown_error"

# ==============================
# Parallel batch processing
# ==============================
rows = []
batch_num = start_batch

# Convert the streaming iterable into indexed tuples once.
# Still memory-light because we only hold indices and refs, but to be safe,
# iterate and handle resume skipping on the fly.
def iter_with_index(iterable, start=1):
    for i, s in enumerate(iterable, start=start):
        yield i, s

current_window = []
window_start = start_index + 1
next_cutoff = start_index + BATCH_SIZE

for i, sample in iter_with_index(subset, start=1):
    # Skip until resume point
    if i <= start_index:
        continue

    current_window.append((i, sample, batch_num))
    # If window full, process in parallel
    if i >= next_cutoff:
        # Parallel execution
        results = []
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(fetch_and_save, item): item[0] for item in current_window}
            for fut in as_completed(futures):
                idx, record, err = fut.result()
                if record:
                    rows.append(record)
                else:
                    print(f"[{idx}] Failed: {err}")

        # Save batch
        print('Creating DataFrame and saving to Excel...')
        df = pd.DataFrame(rows)
        out_file = f"{EXCEL_PREFIX}_batch{batch_num}.xlsx"
        df.to_excel(out_file, index=False)
        print(f"✅ Saved batch {batch_num} with {len(df)} samples → {out_file}")

        # Prepare next batch
        rows = []
        batch_num += 1
        current_window = []
        next_cutoff += BATCH_SIZE

# Process leftovers
if current_window:
    results = []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fetch_and_save, item): item[0] for item in current_window}
        for fut in as_completed(futures):
            idx, record, err = fut.result()
            if record:
                rows.append(record)
            else:
                print(f"[{idx}] Failed: {err}")
    print('Creating DataFrame and saving to Excel...')
    df = pd.DataFrame(rows)
    out_file = f"{EXCEL_PREFIX}_batch{batch_num}.xlsx"
    df.to_excel(out_file, index=False)
    print(f"✅ Saved final batch {batch_num} with {len(df)} samples → {out_file}")
