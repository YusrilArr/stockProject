"""Rename file, validasi, dan ekspor CSV sesuai format impor Adobe Stock."""
from __future__ import annotations

import csv
import io
import re
import unicodedata
import zipfile
from datetime import datetime
from pathlib import Path

from core.config import ROOT_DIR
from metadata.generator import CATEGORIES, KEYWORDS_MAX, TITLE_MAX, ImageMeta

FILENAME_MAX = 30   # batas Adobe, termasuk ekstensi
CSV_HEADER = ["Filename", "Title", "Keywords", "Category", "Releases"]
CSV_NAME = "adobe_stock_metadata.csv"
OUTPUT_DIR = ROOT_DIR / "output" / "metadata"


def slugify(text: str) -> str:
    t = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", t.lower()).strip("-")


def make_filename(title: str, ext: str, used: set[str]) -> str:
    """Nama file dari judul: huruf kecil, pakai tanda hubung, ≤30 karakter termasuk ekstensi, unik."""
    ext = "." + ext.lower().lstrip(".")
    room = FILENAME_MAX - len(ext)
    slug = slugify(title) or "image"
    if len(slug) > room:
        cut = slug[:room + 1].rsplit("-", 1)[0]
        slug = cut if len(cut) >= room // 2 else slug[:room].rstrip("-")
    name, n = slug + ext, 1
    while name.lower() in used:
        n += 1
        suffix = f"-{n}"
        name = slug[:room - len(suffix)].rstrip("-") + suffix + ext
    used.add(name.lower())
    return name


def assign_filenames(items: list[ImageMeta]) -> None:
    used: set[str] = set()
    for it in items:
        it.filename = make_filename(it.title, Path(it.source).suffix or ".jpg", used)


def validate(items: list[ImageMeta]) -> tuple[list[str], list[str]]:
    """Return (errors, warnings).

    errors: CSV akan ditolak atau file tidak cocok, jadi ekspor dikunci.
    warnings: masih boleh diekspor, tapi sebaiknya diperbaiki.
    """
    errors, warnings, seen = [], [], set()
    for it in items:
        label = it.filename or it.source
        if not it.filename:
            errors.append(f"{it.source}: nama file kosong.")
        elif len(it.filename) > FILENAME_MAX:
            errors.append(f"{label}: nama file {len(it.filename)} karakter, maksimal {FILENAME_MAX}.")
        if Path(it.filename).suffix.lower() != Path(it.source).suffix.lower():
            errors.append(f"{label}: ekstensi harus sama dengan file aslinya ({Path(it.source).suffix}).")
        if it.filename.lower() in seen:
            errors.append(f"{label}: nama file dipakai lebih dari sekali.")
        seen.add(it.filename.lower())
        if not it.title:
            errors.append(f"{label}: judul kosong.")
        elif len(it.title) > TITLE_MAX:
            warnings.append(f"{label}: judul {len(it.title)} karakter, disarankan maksimal {TITLE_MAX}.")
        if "," in it.title:
            errors.append(f"{label}: judul tidak boleh memakai koma.")
        if not it.keywords:
            errors.append(f"{label}: keyword kosong.")
        elif len(it.keywords) > KEYWORDS_MAX:
            errors.append(f"{label}: {len(it.keywords)} keyword, maksimal {KEYWORDS_MAX}.")
        if it.category not in CATEGORIES:
            warnings.append(f"{label}: kategori belum dipilih (opsional, tapi sangat disarankan).")
    return errors, warnings


def build_csv(items: list[ImageMeta]) -> str:
    """CSV UTF-8 tanpa BOM, baris diakhiri LF — sama dengan file contoh resmi Adobe."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_HEADER)
    for it in items:
        w.writerow([it.filename, it.title, ", ".join(it.keywords), it.category or "", ""])
    return buf.getvalue()


def save_package(items: list[ImageMeta], files: dict[str, bytes], root: Path = OUTPUT_DIR) -> Path:
    """Simpan file yang sudah di-rename + CSV ke satu folder baru."""
    folder = root / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    for it in items:
        (folder / it.filename).write_bytes(files[it.source])
    (folder / CSV_NAME).write_text(build_csv(items), encoding="utf-8", newline="")
    return folder


def zip_package(items: list[ImageMeta], files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        for it in items:
            z.writestr(it.filename, files[it.source])
        z.writestr(CSV_NAME, build_csv(items))
    return buf.getvalue()
