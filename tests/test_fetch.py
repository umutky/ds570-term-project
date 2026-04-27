import hashlib
from pathlib import Path

import pytest

from retail_forecast.data.fetch import fetch


def test_cache_hit_skips_download(tmp_path):
    """If the file already exists, fetch() returns it without touching the network."""
    dest = tmp_path / "m5_ca1_subset.parquet"
    dest.write_bytes(b"fake parquet data")

    result = fetch(url="http://should-not-be-called.invalid", dest=dest, expected_sha256="")
    assert result == dest


def test_hash_mismatch_raises(tmp_path, monkeypatch):
    """fetch() should raise ValueError and delete the file when the SHA256 is wrong."""
    dest = tmp_path / "m5_ca1_subset.parquet"

    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(b"wrong content")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    correct_hash = hashlib.sha256(b"correct content").hexdigest()
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        fetch(url="http://example.invalid", dest=dest, expected_sha256=correct_hash)

    assert not dest.exists(), "fetch() must delete the file on hash mismatch"


def test_hash_match_succeeds(tmp_path, monkeypatch):
    """fetch() should succeed and return dest when SHA256 matches."""
    dest = tmp_path / "m5_ca1_subset.parquet"
    content = b"valid parquet content"

    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(content)

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    expected = hashlib.sha256(content).hexdigest()
    result = fetch(url="http://example.invalid", dest=dest, expected_sha256=expected)
    assert result == dest
    assert dest.exists()
