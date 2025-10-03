import os
import glob
import time
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
BATCH_SIZE = 2000                 # how many per batch (when batching)
MAX_SAMPLES = 10000000          # total samples desired
EXCEL_PREFIX = "laion_subset"     # Excel base name
WORKERS = 32                      # parallel downloads per batch
REQ_TIMEOUT = 6                   # seconds per request
MAX_RETRIES = 2                   # per-URL retry attempts (in addition to pool retries)
BACKOFF_BASE = 0.5                # backoff base seconds

# New toggles
SINGLE_FILE_MODE = True           # if True, disable batching and save one Excel
FLUSH_EVERY = 5000                # rows per flush when SINGLE_FILE_MODE

os.makedirs(OUT_DIR, exist_ok=True)

# ==============================
# Resume logic
# ==============================
if SINGLE_FILE_MODE:
    single_excel = f"{EXCEL_PREFIX}.xlsx"
    if os.path.isfile(single_excel):
        print(f"➡️ Single-file resume from {single_excel}")
        try:
            existing_df = pd.read_excel(single_excel)
            if "image_path" in existing_df.columns:
                already_done = set(existing_df["image_path"].astype(str).tolist())
            else:
                already_done = set()
        except Exception as e:
            print(f"⚠️ Could not read existing single Excel, starting fresh: {e}")
            already_done = set()
    else:
        print("➡️ Single-file mode, starting fresh")
        already_done = set()
    # Single-file mode uses batch_num=1 for naming consistency, but filenames differ by index.
    start_batch = 1
    start_index = 0
else:
    # Per-batch resume logic (unchanged)
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
        total=3,
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
    sess.headers.update({"User-Agent": "Mozilla/5.0 (compatible; laion-downloader/1.0)"})
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

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQ_TIMEOUT, stream=True)
            if resp.status_code != 200 or resp.content is None:
                raise RuntimeError(f"bad_status:{resp.status_code}")

            content = resp.content
            if len(content) > 30 * 1024 * 1024:
                raise RuntimeError("too_large")

            with Image.open(BytesIO(content)) as im:
                im = im.convert("RGB")
                # In single-file mode, keep a consistent, deterministic filename
                # using global index; in batch mode, keep existing name pattern.
                if SINGLE_FILE_MODE:
                    img_path = os.path.join(OUT_DIR, f"sample_{idx}.jpg")
                else:
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
# Helpers
# ==============================
def iter_with_index(iterable, start=1):
    for i, s in enumerate(iterable, start=start):
        yield i, s

def flush_append_excel(out_file, rows_df, exists):
    # If file does not exist, write header; else append below existing
    if not os.path.isfile(out_file) or not exists:
        rows_df.to_excel(out_file, index=False)
    else:
        # Append by reading, concatenating, and rewriting (simplest, avoids engine issues)
        base = pd.read_excel(out_file)
        out = pd.concat([base, rows_df], ignore_index=True)
        out.to_excel(out_file, index=False)

# ==============================
# Main processing
# ==============================
rows = []
batch_num = start_batch

current_window = []
next_cutoff = start_index + (BATCH_SIZE if not SINGLE_FILE_MODE else max(BATCH_SIZE, FLUSH_EVERY))

processed = 0
for i, sample in iter_with_index(subset, start=1):
    if not SINGLE_FILE_MODE and i <= start_index:
        continue

    # In single-file mode, optionally skip items whose image already exists in Excel
    if SINGLE_FILE_MODE and 'already_done' in globals():
        # Predict the image path to check presence
        predicted_path = os.path.join(OUT_DIR, f"sample_{i}.jpg")
        if predicted_path in already_done:
            continue

    current_window.append((i, sample, batch_num))

    # Decide window boundary:
    boundary = (i >= next_cutoff)
    if boundary:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(fetch_and_save, item): item[0] for item in current_window}
            for fut in as_completed(futures):
                idx, record, err = fut.result()
                if record:
                    rows.append(record)
                else:
                    print(f"[{idx}] Failed: {err}")

        # Save results
        if SINGLE_FILE_MODE:
            out_file = f"{EXCEL_PREFIX}.xlsx"
            print('Flushing to single Excel...')
            df = pd.DataFrame(rows)
            file_exists = os.path.isfile(out_file)
            flush_append_excel(out_file, df, file_exists)
            # Update already_done for further skips
            if 'already_done' in globals():
                for p in df['image_path']:
                    already_done.add(str(p))
            print(f"✅ Flushed {len(df)} rows → {out_file}")
            rows = []
            current_window = []
            next_cutoff = i + FLUSH_EVERY
        else:
            print('Creating DataFrame and saving to Excel...')
            df = pd.DataFrame(rows)
            out_file = f"{EXCEL_PREFIX}_batch{batch_num}.xlsx"
            df.to_excel(out_file, index=False)
            print(f"✅ Saved batch {batch_num} with {len(df)} samples → {out_file}")
            rows = []
            batch_num += 1
            current_window = []
            next_cutoff += BATCH_SIZE

# Process leftovers
if current_window:
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(fetch_and_save, item): item[0] for item in current_window}
        for fut in as_completed(futures):
            idx, record, err = fut.result()
            if record:
                rows.append(record)
            else:
                print(f"[{idx}] Failed: {err}")

    df = pd.DataFrame(rows)
    if SINGLE_FILE_MODE:
        out_file = f"{EXCEL_PREFIX}.xlsx"
        print('Final flush to single Excel...')
        file_exists = os.path.isfile(out_file)
        flush_append_excel(out_file, df, file_exists)
        if 'already_done' in globals():
            for p in df['image_path']:
                already_done.add(str(p))
        print(f"✅ Saved final {len(df)} rows → {out_file}")
    else:
        out_file = f"{EXCEL_PREFIX}_batch{batch_num}.xlsx"
        print('Creating DataFrame and saving to Excel...')
        df.to_excel(out_file, index=False)
        print(f"✅ Saved final batch {batch_num} with {len(df)} samples → {out_file}")
