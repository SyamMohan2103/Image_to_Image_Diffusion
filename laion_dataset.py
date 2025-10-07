import glob
import os
from typing import List, Optional
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image, UnidentifiedImageError


class ImageCaptionDataset(Dataset):
    """
    Dataset that reads `image_path, caption` from one or more files.

    Usage options:
    - Provide a directory and a prefix; it will load all files matching
      f\"{prefix}_batch*.xlsx\" and f\"{prefix}_batch*.csv\".
    - Or provide an explicit list of files in `files`.

    Required columns: `image_path` and `caption` (customizable via args).

    Returns: (PIL.Image.Image, caption_str)
    """

    def __init__(
        self,
        dir_or_single_csv: Optional[str] = None,
        *,
        prefix: Optional[str] = None,
        files: Optional[List[str]] = None,
        image_col: str = "image_path",
        caption_col: str = "caption",
        drop_na: bool = True,
        dedupe: bool = True,
        on_image_error: str = "raise",  # "raise" | "skip"
    ):
        """
        Usage
            Load discovered Excel batches:
            dataset = ImageCaptionDataset(".", prefix="laion_subset")

            Load from image directory with batch files elsewhere:
            dataset = ImageCaptionDataset("/path/to/batches", prefix="laion_subset")

            Load explicit file list:
            dataset = ImageCaptionDataset(files=["laion_subset_batch1.xlsx","laion_subset_batch2.xlsx"])

        Args:
            dir_or_single_csv: Either a directory to scan or a single CSV/XLSX file path.
            prefix: If scanning a directory, file prefix like 'laion_subset' to match batches.
            files: Explicit list of CSV/XLSX files to load (overrides directory scan).
            image_col: Column name for image path.
            caption_col: Column name for caption.
            drop_na: Drop rows where required cols are NA/empty.
            dedupe: Drop exact duplicate rows by [image_col, caption_col].
            on_image_error: If 'skip', IOError/UnidentifiedImageError will be skipped on __getitem__.
        """
        if files is None:
            if dir_or_single_csv is None:
                raise ValueError("Provide either files=[...] or dir_or_single_csv")
            if os.path.isdir(dir_or_single_csv):
                if not prefix:
                    raise ValueError("When a directory is provided, 'prefix' is required to discover batch files.")
                # Discover both Excel and CSV batches
                xlsx = sorted(glob.glob(os.path.join(dir_or_single_csv, f"{prefix}_batch*.xlsx")))
                csvs = sorted(glob.glob(os.path.join(dir_or_single_csv, f"{prefix}_batch*.csv")))
                discovered = xlsx + csvs
                if not discovered:
                    raise FileNotFoundError(f"No batch files found matching {prefix}_batch*.xlsx or .csv in {dir_or_single_csv}")
                files = discovered
            else:
                # Single file path
                files = [dir_or_single_csv]

        # Load and concatenate
        frames = []
        for fp in files:
            ext = os.path.splitext(fp)[1].lower()
            if ext in (".xlsx", ".xls"):
                df = pd.read_excel(fp)
            elif ext == ".csv":
                df = pd.read_csv(fp)
            else:
                raise ValueError(f"Unsupported file type: {fp}")
            if image_col not in df.columns or caption_col not in df.columns:
                raise ValueError(f"File {fp} must contain columns '{image_col}' and '{caption_col}'")
            frames.append(df[[image_col, caption_col]])

        if not frames:
            raise ValueError("No data loaded from provided files.")

        df = pd.concat(frames, axis=0, ignore_index=True)

        # Clean up
        if drop_na:
            df = df.dropna(subset=[image_col, caption_col])
        # Normalize to str for caption
        df[caption_col] = df[caption_col].astype(str)
        df[image_col] = df[image_col].str.replace(r'\\', '/', regex=True) # Converting windows paths to unix style for consistency

        if dedupe:
            df = df.drop_duplicates(subset=[image_col, caption_col]).reset_index(drop=True)

        # Optional: filter out non-existing files to avoid runtime errors
        # Keep rows with existing files; others will be handled at __getitem__
        self._exists_mask = df[image_col].map(lambda p: isinstance(p, str) and os.path.isfile(p))
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.caption_col = caption_col
        self.on_image_error = on_image_error

        # Build an index of valid rows (existing files first) to avoid many open failures
        self._indices = [i for i, ok in enumerate(self._exists_mask) if ok]

    def __len__(self):
        # Only count rows for which files currently exist
        return len(self._indices)

    def __getitem__(self, idx: int):
        real_idx = self._indices[idx]
        row = self.df.iloc[real_idx]
        img_path = row[self.image_col]
        caption = str(row[self.caption_col])
        try:
            img = Image.open(img_path).convert("RGB")
            if img.width < 5 or img.height < 5:
                print(f"Invalid image dimensions at {img_path}: {img.size}")
                raise IndexError(f"Skipped invalid image at {img_path}: {img.size}")    
            return img, caption
        except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
            if self.on_image_error == "skip":
                # Fall back: try next valid index by raising IndexError to trigger DataLoader re-sample if using sampler
                raise IndexError(f"Skipped corrupted/missing image at {img_path}: {e}")
            raise
