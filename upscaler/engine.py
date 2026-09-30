"""Mesin upscale: Real-ESRGAN via PyTorch (CPU saja), dengan cadangan Lanczos.

torch & spandrel di-import saat dibutuhkan saja, supaya aplikasi tetap cepat dibuka
dan modul lain tidak ikut terbebani.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from core.config import ROOT_DIR

MODELS_DIR = ROOT_DIR / "models"
_GH = "https://github.com/xinntao/Real-ESRGAN/releases/download"


@dataclass(frozen=True)
class ModelInfo:
    label: str
    filename: str
    url: str
    sha256: str


# Semua model memperbesar 4×. Hasilnya lalu diperkecil ke ukuran target (lihat pipeline.py).
MODELS = {
    "photo_fast": ModelInfo(
        "Foto — cepat (realesr-general-x4v3)", "realesr-general-x4v3.pth",
        f"{_GH}/v0.2.5.0/realesr-general-x4v3.pth",
        "8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292",
    ),
    "photo_best": ModelInfo(
        "Foto — maksimal (RealESRGAN_x4plus)", "RealESRGAN_x4plus.pth",
        f"{_GH}/v0.1.0/RealESRGAN_x4plus.pth",
        "4fa0d38905f75ac06eb49a7951b426670021be3018265fd191d2125df9d682f1",
    ),
    "icon_fast": ModelInfo(
        "Icon — cepat (realesr-animevideov3)", "realesr-animevideov3.pth",
        f"{_GH}/v0.2.5.0/realesr-animevideov3.pth",
        "b8a8376811077954d82ca3fcf476f1ac3da3e8a68a4f4d71363008000a18b75d",
    ),
    "icon_best": ModelInfo(
        "Icon — maksimal (RealESRGAN_x4plus_anime_6B)", "RealESRGAN_x4plus_anime_6B.pth",
        f"{_GH}/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da",
    ),
}

MODEL_SCALE = 4
# Model besar makan RAM ±3,7GB dengan tile 512; tile 256 menekan RAM dengan sedikit tambahan waktu.
TILE = {"photo_fast": 512, "icon_fast": 512, "photo_best": 256, "icon_best": 256}
_loaded: dict[str, object] = {}


class EngineError(Exception):
    """Mesin AI tidak bisa dipakai; pesan aman ditampilkan ke user."""


def engine_available() -> tuple[bool, str]:
    """Cek apakah torch & spandrel terpasang, tanpa memuatnya (memuat torch butuh beberapa detik).

    Return (ok, pesan).
    """
    for name in ("torch", "spandrel"):
        if importlib.util.find_spec(name) is None:
            return False, f"Library upscale AI belum terpasang ({name}). Jalankan: pip install -r requirements.txt"
    return True, "ok"


def default_threads(save_power: bool) -> int:
    """Jumlah thread CPU. Mode hemat memakai separuh core fisik."""
    try:
        import psutil
        physical = psutil.cpu_count(logical=False) or 4
    except ImportError:
        physical = max((os.cpu_count() or 4) // 2, 1)
    return max(physical // 2, 1) if save_power else physical


def model_path(key: str) -> Path:
    return MODELS_DIR / MODELS[key].filename


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(key: str, progress: Callable[[float], None] | None = None) -> Path:
    """Unduh file model sekali saja dari rilis resmi Real-ESRGAN, lalu cek hash-nya."""
    info = MODELS[key]
    path = model_path(key)
    if path.exists():
        return path
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    try:
        with urllib.request.urlopen(info.url, timeout=60) as resp, open(tmp, "wb") as f:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1 << 16):
                f.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(done / total)
    except OSError as e:
        tmp.unlink(missing_ok=True)
        raise EngineError(
            f"Gagal mengunduh model {info.filename}. Cek koneksi internet "
            "(jaringan kantor mungkin memblokir GitHub), lalu coba lagi."
        ) from e
    if _sha256(tmp) != info.sha256:
        tmp.unlink(missing_ok=True)
        raise EngineError(f"File model {info.filename} rusak saat diunduh. Coba lagi.")
    tmp.replace(path)
    return path


def _load(key: str):
    if key not in _loaded:
        from spandrel import ModelLoader

        descriptor = ModelLoader().load_from_file(str(ensure_model(key)))
        if descriptor.scale != MODEL_SCALE:
            raise EngineError(f"Model {key} bukan model 4×.")
        _loaded[key] = descriptor.model.eval()
    return _loaded[key]


def upscale_rgb(
    arr: np.ndarray,
    key: str,
    *,
    threads: int,
    tile: int | None = None,
    pad: int = 16,
    progress: Callable[[float], None] | None = None,
) -> np.ndarray:
    """Perbesar array RGB uint8 (H, W, 3) sebanyak 4× dengan model AI.

    Gambar diproses per potongan (tile) agar pemakaian RAM tetap kecil.
    Setiap potongan diberi pinggiran (pad) supaya sambungannya tidak terlihat.
    """
    ok, msg = engine_available()
    if not ok:
        raise EngineError(msg)
    import torch

    torch.set_num_threads(threads)
    model = _load(key)
    tile = tile or TILE.get(key, 512)
    s = MODEL_SCALE
    x = torch.from_numpy(np.array(arr, dtype=np.uint8)).permute(2, 0, 1).float().div_(255).unsqueeze(0)
    _, c, h, w = x.shape
    out = torch.empty(1, c, h * s, w * s)
    tiles = [(y, x0) for y in range(0, h, tile) for x0 in range(0, w, tile)]
    with torch.inference_mode():
        for i, (y, x0) in enumerate(tiles, 1):
            y1, x1 = max(y - pad, 0), max(x0 - pad, 0)
            y2, x2 = min(y + tile + pad, h), min(x0 + tile + pad, w)
            res = model(x[:, :, y1:y2, x1:x2])
            th, tw = min(tile, h - y) * s, min(tile, w - x0) * s
            oy, ox = (y - y1) * s, (x0 - x1) * s
            out[:, :, y * s:y * s + th, x0 * s:x0 * s + tw] = res[:, :, oy:oy + th, ox:ox + tw]
            if progress:
                progress(i / len(tiles))
    return out.squeeze(0).clamp_(0, 1).mul_(255).round_().byte().permute(1, 2, 0).numpy()
