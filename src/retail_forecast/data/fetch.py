import hashlib
import urllib.request
from pathlib import Path

from retail_forecast.config import DATA_SHA256, DATA_URL, RAW_DATA_DIR, SUBSET_FILE


def fetch(
    url: str = DATA_URL,
    dest: Path | None = None,
    expected_sha256: str = DATA_SHA256,
) -> Path:
    """Download the CA_1 subset parquet from GitHub Release if not cached locally.

    On first call the file is downloaded and its SHA256 is verified.
    Subsequent calls return the cached file immediately.

    Args:
        url: Direct download URL (defaults to config.DATA_URL).
        dest: Local destination path (defaults to data/raw/m5_ca1_subset.parquet).
        expected_sha256: Expected hex digest; skipped when empty string.

    Returns:
        Path to the local parquet file.
    """
    if dest is None:
        dest = RAW_DATA_DIR / SUBSET_FILE

    if dest.exists():
        print(f"Cache hit: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)

    if expected_sha256:
        actual = hashlib.sha256(dest.read_bytes()).hexdigest()
        if actual != expected_sha256:
            dest.unlink()
            raise ValueError(
                f"SHA256 mismatch.\n  expected: {expected_sha256}\n  got:      {actual}"
            )
        print("SHA256 verified.")

    print(f"Saved to {dest}")
    return dest
