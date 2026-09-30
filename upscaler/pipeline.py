"""Alur lengkap Modul 2: buka → sRGB → (pangkas icon) → upscale → simpan JPEG/PNG."""
from __future__ import annotations

import io
import re
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageCms, ImageOps

from core.config import ROOT_DIR
from upscaler import engine
from upscaler.detect import ICON, PHOTO, has_alpha
from upscaler.icon import ALPHA_VISIBLE, autocrop, bleed_colors
from upscaler.plan import TARGET_MP, make_plan

MAX_FILE_MB = 45
MAX_BATCH = 8
OUTPUT_DIR = ROOT_DIR / "output" / "upscale"
SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

Progress = Callable[[float, str], None]


@dataclass
class Result:
    filename: str
    kind: str
    src_size: tuple[int, int]
    out_size: tuple[int, int]
    engine: str
    seconds: float
    data: bytes
    source: str = ""   # nama file asli yang di-upload
    warnings: list[str] = field(default_factory=list)

    @property
    def mime(self) -> str:
        return "image/png" if self.filename.endswith(".png") else "image/jpeg"

    @property
    def size_mb(self) -> float:
        return len(self.data) / 1e6


# ---------- membuka & warna ----------

def to_srgb(img: Image.Image) -> tuple[Image.Image, str | None]:
    """Konversi gambar ke RGB/RGBA sRGB. Return (gambar, catatan konversi atau None)."""
    note = None
    icc = img.info.get("icc_profile")
    alpha = img.getchannel("A") if has_alpha(img) else None
    if alpha is not None and img.mode != "RGBA":
        img = img.convert("RGBA")
        alpha = img.getchannel("A")
    if icc:
        try:
            src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
            desc = ImageCms.getProfileDescription(src).strip()
            if "srgb" not in desc.lower():
                base = img.convert("RGB") if img.mode not in ("RGB", "CMYK") else img
                img = ImageCms.profileToProfile(base, src, ImageCms.createProfile("sRGB"), outputMode="RGB")
                note = f"Profil warna '{desc}' dikonversi ke sRGB."
        except (OSError, ImageCms.PyCMSError):
            note = "Profil warna di file tidak terbaca; dianggap sRGB."
    elif img.mode == "CMYK":
        note = "Gambar CMYK tanpa profil; dikonversi ke RGB secara sederhana."
    rgb = img.convert("RGB")
    if alpha is not None:
        rgb.putalpha(alpha)
    return rgb, note


def open_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as e:  # noqa: BLE001
        raise ValueError("File bukan gambar yang bisa dibaca (gunakan PNG, JPEG, atau WebP).") from e
    icc = img.info.get("icc_profile")
    img = ImageOps.exif_transpose(img)
    if icc:
        img.info["icc_profile"] = icc
    return img


# ---------- upscale ----------

def _resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize(size, Image.LANCZOS))


def _ai_upscale(arr, plan, key, threads, report) -> np.ndarray:
    """Jalankan model 4× sebanyak plan.passes, perkecil ke target di akhir.

    Di antara dua putaran, gambar diperkecil dulu ke ukuran yang dibutuhkan,
    supaya putaran kedua tidak menghasilkan gambar raksasa.
    """
    for p in range(plan.passes):
        arr = engine.upscale_rgb(arr, key, threads=threads,
                                 progress=lambda f, p=p: report((p + f) / plan.passes))
        remaining = plan.passes - p - 1
        if remaining:
            div = engine.MODEL_SCALE ** remaining
            arr = _resize(arr, (-(-plan.target_w // div), -(-plan.target_h // div)))
    return _resize(arr, (plan.target_w, plan.target_h))


def process(
    data: bytes,
    name: str,
    kind: str,
    *,
    quality: str = "fast",
    threads: int = 4,
    target_mp: float = TARGET_MP,
    progress: Progress | None = None,
) -> Result:
    """Proses satu gambar. quality: 'fast' atau 'best'."""
    t0 = time.time()
    warnings: list[str] = []

    def report(frac: float, msg: str = "Upscale...") -> None:
        if progress:
            progress(min(max(frac, 0.0), 1.0), msg)

    img, note = to_srgb(open_image(data))
    if note:
        warnings.append(note)

    if kind == ICON:
        if img.mode != "RGBA":
            img = img.convert("RGBA")
            warnings.append("Background tidak transparan. Adobe mewajibkan PNG icon ber-background "
                            "transparan; hapus background-nya dulu.")
        else:
            before = img.size
            img = autocrop(img)
            if img.size != before:
                warnings.append(f"Ruang kosong dipangkas: {before[0]}×{before[1]} → {img.width}×{img.height}.")
    elif img.mode == "RGBA":
        if np.asarray(img.getchannel("A")).min() < 255:
            warnings.append("Bagian transparan diganti putih karena foto disimpan sebagai JPEG.")
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.getchannel("A"))
        img = bg

    plan = make_plan(img.width, img.height, target_mp)
    used = "Tidak perlu upscale (sudah ≥4MP)"
    if plan.needs_upscale:
        key = f"{'icon' if kind == ICON else 'photo'}_{quality}"
        arr = np.asarray(img)
        try:
            if kind == ICON:
                rgb = bleed_colors(arr)
                a3 = np.repeat(arr[..., 3:4], 3, axis=2)
                rgb_up = _ai_upscale(rgb, plan, key, threads, lambda f: report(f / 2, "Upscale warna..."))                # Transparansi hanya masker; model icon cepat sudah cukup dan jauh lebih ringan.
                a_up = _ai_upscale(a3, plan, "icon_fast", threads,
                                   lambda f: report(0.5 + f / 2, "Upscale transparansi..."))
                alpha = a_up.mean(axis=2)
                alpha[alpha < ALPHA_VISIBLE] = 0
                alpha[alpha > 255 - ALPHA_VISIBLE] = 255
                out = np.dstack([rgb_up, alpha.round().astype(np.uint8)])
            else:
                out = _ai_upscale(arr, plan, key, threads, report)
            used = f"AI: {engine.MODELS[key].label}"
        except Exception as e:  # noqa: BLE001 — apa pun error-nya, tetap hasilkan file lewat cadangan
            msg = str(e) if isinstance(e, engine.EngineError) else f"{type(e).__name__}: {e}"
            warnings.append(f"Upscale AI gagal ({msg}). Memakai resize Lanczos, kualitas lebih rendah.")
            out = _resize(arr, (plan.target_w, plan.target_h))
            used = "Lanczos (cadangan)"
        img = Image.fromarray(out)

    report(1.0, "Menyimpan...")
    data_out, ext = encode(img, kind, warnings)
    return Result(
        filename=f"{safe_stem(name)}{ext}", kind=kind, src_size=(plan.width, plan.height),
        out_size=img.size, engine=used, seconds=time.time() - t0, data=data_out, source=name,
        warnings=warnings,
    )


# ---------- simpan ----------

def encode(img: Image.Image, kind: str, warnings: list[str]) -> tuple[bytes, str]:
    """Simpan dengan profil sRGB. Foto → JPEG, icon → PNG. Pastikan ≤45MB."""
    buf = io.BytesIO()
    if kind == ICON:
        img.save(buf, "PNG", icc_profile=SRGB_ICC, compress_level=6)
        if buf.tell() > MAX_FILE_MB * 1e6:
            warnings.append(f"File PNG {buf.tell() / 1e6:.0f}MB melebihi batas Adobe {MAX_FILE_MB}MB.")
        return buf.getvalue(), ".png"
    for q in (95, 92, 88, 85):
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=q, subsampling=0, optimize=True, icc_profile=SRGB_ICC)
        if buf.tell() <= MAX_FILE_MB * 1e6:
            if q < 95:
                warnings.append(f"Kualitas JPEG diturunkan ke {q} agar ≤{MAX_FILE_MB}MB.")
            return buf.getvalue(), ".jpg"
    warnings.append(f"File JPEG tetap di atas {MAX_FILE_MB}MB.")
    return buf.getvalue(), ".jpg"


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return stem or "gambar"


def unique_names(results: list[Result]) -> None:
    """Tambahkan akhiran -2, -3 ... bila ada nama file yang sama."""
    used: set[str] = set()
    for r in results:
        base, ext = r.filename.rsplit(".", 1)
        name, n = r.filename, 1
        while name.lower() in used:
            n += 1
            name = f"{base}-{n}.{ext}"
        r.filename = name
        used.add(name.lower())


def save_results(results: list[Result], root: Path = OUTPUT_DIR) -> Path:
    folder = root / datetime.now().strftime("%Y%m%d-%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    for r in results:
        (folder / r.filename).write_bytes(r.data)
    return folder


def zip_results(results: list[Result]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:  # JPEG/PNG sudah terkompresi
        for r in results:
            z.writestr(r.filename, r.data)
    return buf.getvalue()


__all__ = ["ICON", "PHOTO", "MAX_BATCH", "Result", "process", "save_results", "zip_results", "unique_names"]
