"""Tar archive discovery, checksum verification, and extraction for raw Ali-CCP.

Ali-CCP ships as two tar archives (`sample_train`, `sample_test`) from the Tianchi
dataset page (https://tianchi.aliyun.com/dataset/408). The archive extension varies:
Tianchi's own Data List page currently distributes `sample_train.tar.gz` /
`sample_test.tar.gz`, while some mirrors / older references use the bare `.tar` name
(see `data.py`'s `SAMPLE_TRAIN_TAR` / `SAMPLE_TEST_TAR` constants, which assume no
`.gz`). Discovery here accepts `.tar`, `.tar.gz`, and `.tgz` so either works.

This module is new: no prior version of this pipeline checksummed the archive before
extracting it. If Tianchi (or your mirror) publishes an MD5 file alongside the tarball
(e.g. `sample_train.tar.gz.md5` next to `sample_train.tar.gz`), drop it in the same
directory and `extract_archives` will verify it before extracting. The checksum file is
optional and best-effort — if it isn't there, extraction proceeds with a warning rather
than failing, since not every mirror publishes one.
"""
from __future__ import annotations

import hashlib
import os
import tarfile

# Base names (without extension) for the two Ali-CCP sample archives, and the raw CSVs
# used to detect "already extracted". Duplicated from data.py's constants (rather than
# imported) so this module has no dependency on data.py and can be used on its own.
SAMPLE_TRAIN_BASENAME = "sample_train"
SAMPLE_TEST_BASENAME = "sample_test"
ARCHIVE_EXTENSIONS = (".tar.gz", ".tgz", ".tar")

SAMPLE_SKELETON_TRAIN = "sample_skeleton_train.csv"
COMMON_FEATURES_TRAIN = "common_features_train.csv"
SAMPLE_SKELETON_TEST = "sample_skeleton_test.csv"
COMMON_FEATURES_TEST = "common_features_test.csv"

MD5_CHUNK_BYTES = 8 * 1024 * 1024  # 8 MiB — archives are multi-GB; never load whole file


def _find_file_recursive(root, filename):
    """Find filename under root (direct child or any subdirectory)."""
    direct = os.path.join(root, filename)
    if os.path.isfile(direct):
        return direct
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def find_archive(data_dir, basename):
    """Locate `<basename><ext>` under data_dir, trying each of ARCHIVE_EXTENSIONS.

    Returns the first match (direct child or any subdirectory), or None if the archive
    isn't present under data_dir in any of the extension forms.
    """
    for ext in ARCHIVE_EXTENSIONS:
        path = _find_file_recursive(data_dir, basename + ext)
        if path:
            return path
    return None


def compute_md5(path, chunk_bytes=MD5_CHUNK_BYTES):
    """Stream `path` through hashlib.md5 in chunks; returns the hex digest."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_md5(archive_path):
    """Best-effort MD5 check against a co-located `<archive_path>.md5` file.

    If `<archive_path>.md5` exists, its contents are compared against the archive's
    computed MD5 and a ValueError is raised on mismatch (checksum files are commonly
    "<hex>" or "<hex>  <filename>"; only the first whitespace-separated token is used).
    If no `.md5` file is present, this is a no-op (prints a warning) rather than a hard
    failure — the checksum file is optional and not every mirror provides one.
    """
    md5_path = archive_path + ".md5"
    if not os.path.isfile(md5_path):
        print(f"[extract] no checksum file at {md5_path}; skipping MD5 verification for {archive_path}")
        return
    with open(md5_path, "r") as f:
        expected = f.read().strip().split()[0].lower()
    actual = compute_md5(archive_path).lower()
    if actual != expected:
        raise ValueError(
            f"MD5 mismatch for {archive_path}: expected {expected}, got {actual}. "
            "Re-download the archive — it is likely corrupted or truncated."
        )
    print(f"[extract] MD5 verified for {archive_path}")


def _already_extracted(data_dir):
    return (
        _find_file_recursive(data_dir, SAMPLE_SKELETON_TRAIN) is not None
        and _find_file_recursive(data_dir, COMMON_FEATURES_TRAIN) is not None
        and _find_file_recursive(data_dir, SAMPLE_SKELETON_TEST) is not None
        and _find_file_recursive(data_dir, COMMON_FEATURES_TEST) is not None
    )


def extract_archives(data_dir, verify_checksums=True):
    """Locate, (optionally) checksum-verify, and extract the Ali-CCP sample archives.

    Mirrors the has_archives/needs_extract guard used in the original notebook cell
    (experiments/20260404_ali_cpp_esmm/20260404_esmm_experiment.ipynb, cell 6): if all
    four raw CSVs (sample_skeleton_train.csv / common_features_train.csv /
    sample_skeleton_test.csv / common_features_test.csv) are already present anywhere
    under data_dir, extraction is skipped entirely.

    Returns the list of archive paths that were extracted (empty if nothing to do).
    Raises FileNotFoundError if nothing is extracted yet and no archive can be found.
    """
    if _already_extracted(data_dir):
        print(f"[extract] raw CSVs already present under {data_dir}; skipping extraction.")
        return []

    train_path = find_archive(data_dir, SAMPLE_TRAIN_BASENAME)
    test_path = find_archive(data_dir, SAMPLE_TEST_BASENAME)
    if train_path is None or test_path is None:
        missing = []
        if train_path is None:
            missing.append(f"{SAMPLE_TRAIN_BASENAME}(.tar|.tar.gz|.tgz)")
        if test_path is None:
            missing.append(f"{SAMPLE_TEST_BASENAME}(.tar|.tar.gz|.tgz)")
        raise FileNotFoundError(
            f"No extracted CSVs and missing archive(s) under {data_dir}: {', '.join(missing)}. "
            "Download sample_train / sample_test from https://tianchi.aliyun.com/dataset/408 "
            "and place them under data_dir."
        )

    extracted = []
    for path in (train_path, test_path):
        if verify_checksums:
            verify_md5(path)
        print(f"[extract] extracting {path} -> {data_dir} ...")
        with tarfile.open(path, "r:*") as tf:
            # PEP 706 'data' filter (default extraction behavior on Python 3.12+) rejects
            # archive members that would escape data_dir via '../' paths or symlinks;
            # fall back to a plain extractall on pre-3.12 Pythons that lack the kwarg.
            try:
                tf.extractall(data_dir, filter="data")
            except TypeError:
                tf.extractall(data_dir)
        extracted.append(path)
        print(f"[extract] done: {path}")
    return extracted


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data_dir", help="Directory containing the sample_train/sample_test archives")
    p.add_argument("--no-verify", action="store_true", help="Skip MD5 verification even if a .md5 file is present")
    args = p.parse_args()
    extract_archives(args.data_dir, verify_checksums=not args.no_verify)
