"""Deteksi otomatis: foto/ilustrasi atau icon/PNG transparan."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

PHOTO = "Foto / ilustrasi"
ICON = "Icon / PNG transparan"
KINDS = [PHOTO, ICON]

# Ambang batas. Diukur pada gambar yang diperkecil ke maks 256px.
TRANSPARENT_MIN = 0.05   # ≥5% piksel transparan → icon
FEW_COLORS = 3000        # jumlah warna unik (5 bit per channel) di bawah ini → palet sempit
FLAT_MIN = 0.55          # ≥55% piksel berada di area warna rata → gaya flat
VERY_FLAT = 0.80         # sangat rata → icon walau warnanya banyak (anti-aliasing)


@dataclass
class Detection:
    kind: str
    transparent: float   # 0–1
    colors: int
    flat: float          # 0–1

    @property
    def reason(self) -> str:
        return (f"{self.transparent:.0%} transparan · {self.colors:,} warna · "
                f"{self.flat:.0%} area rata").replace(",", ".")


def has_alpha(img: Image.Image) -> bool:
    return img.mode in ("RGBA", "LA", "PA") or (img.mode == "P" and "transparency" in img.info)


def detect(img: Image.Image) -> Detection:
    small = img.copy()
    small.thumbnail((256, 256))
    rgba = np.asarray(small.convert("RGBA"))
    alpha = rgba[..., 3]
    transparent = float((alpha < 16).mean()) if has_alpha(img) else 0.0

    visible = alpha >= 16
    rgb = rgba[..., :3]
    q = (rgb[visible] >> 3).astype(np.uint32)
    colors = int(np.unique((q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]).size) if q.size else 0

    # Area rata: selisih warna dengan tetangga kanan & bawah sangat kecil.
    g = rgb.astype(np.int16)
    dx = np.abs(np.diff(g, axis=1)).sum(-1)[:-1, :]
    dy = np.abs(np.diff(g, axis=0)).sum(-1)[:, :-1]
    flat_mask = (dx + dy) <= 6
    vis = visible[:-1, :-1]
    flat = float(flat_mask[vis].mean()) if vis.any() else 1.0

    is_icon = (
        transparent >= TRANSPARENT_MIN
        or (colors < FEW_COLORS and flat >= FLAT_MIN)
        or flat >= VERY_FLAT
    )
    return Detection(ICON if is_icon else PHOTO, transparent, colors, flat)
