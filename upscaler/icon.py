"""Pengolahan khusus icon / PNG transparan."""
from __future__ import annotations

import numpy as np
from PIL import Image

ALPHA_VISIBLE = 8


def autocrop(img: Image.Image, pad_ratio: float = 0.02) -> Image.Image:
    """Pangkas ruang kosong transparan di sekitar objek, sisakan sedikit pinggiran."""
    alpha = np.asarray(img.getchannel("A"))
    ys, xs = np.nonzero(alpha > ALPHA_VISIBLE)
    if ys.size == 0:
        return img
    pad = round(max(img.size) * pad_ratio)
    box = (
        max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0),
        min(int(xs.max()) + 1 + pad, img.width), min(int(ys.max()) + 1 + pad, img.height),
    )
    return img.crop(box)


def _shift(a: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Geser array 1 piksel tanpa membungkus ke sisi seberang; sisi yang kosong diisi 0."""
    out = np.zeros_like(a)
    h, w = a.shape[:2]
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        a[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


def bleed_colors(rgba: np.ndarray, iterations: int = 24) -> np.ndarray:
    """Isi warna piksel transparan dengan warna tepi objek di dekatnya.

    Piksel transparan sering berwarna hitam/putih. Kalau dibiarkan, model upscale
    ikut "melukis" warna itu ke tepi objek sehingga muncul garis gelap/terang (halo).
    """
    rgb = rgba[..., :3].astype(np.float32)
    known = rgba[..., 3] > ALPHA_VISIBLE
    if known.all() or not known.any():
        return rgba[..., :3].copy()
    rgb[~known] = 0
    for _ in range(iterations):
        if known.all():
            break
        acc = np.zeros_like(rgb)
        cnt = np.zeros(known.shape, np.float32)
        for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            k = _shift(known, dy, dx)
            acc += _shift(rgb, dy, dx) * k[..., None]
            cnt += k
        grow = ~known & (cnt > 0)
        rgb[grow] = acc[grow] / cnt[grow, None]
        known = known | grow
    if not known.all():
        rgb[~known] = rgb[known].mean(0)
    return rgb.round().astype(np.uint8)
