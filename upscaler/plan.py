"""Hitung ukuran target berdasarkan syarat resolusi Adobe Stock."""
from __future__ import annotations

import math
from dataclasses import dataclass

MIN_MP = 4.0      # syarat minimal Adobe untuk JPEG & PNG
TARGET_MP = 4.2   # sedikit di atas minimum supaya aman dari pembulatan
MAX_MP = 100.0
MAX_PASSES = 2    # 2 × model 4× = maks 16×; gambar yang lebih kecil dianggap terlalu kecil


@dataclass
class Plan:
    width: int
    height: int
    target_w: int
    target_h: int
    passes: int   # berapa kali model 4× dijalankan (0 = tidak perlu upscale)

    @property
    def mp(self) -> float:
        return self.width * self.height / 1e6

    @property
    def target_mp(self) -> float:
        return self.target_w * self.target_h / 1e6

    @property
    def factor(self) -> float:
        return self.target_w / self.width

    @property
    def needs_upscale(self) -> bool:
        return self.passes > 0


class PlanError(Exception):
    pass


def make_plan(width: int, height: int, target_mp: float = TARGET_MP) -> Plan:
    if width <= 0 or height <= 0:
        raise PlanError("Ukuran gambar tidak valid.")
    mp = width * height / 1e6
    if mp > MAX_MP:
        raise PlanError(f"Gambar {mp:.0f}MP melebihi batas Adobe {MAX_MP:.0f}MP.")
    if mp >= MIN_MP:
        return Plan(width, height, width, height, 0)

    factor = math.sqrt(target_mp / mp)
    tw, th = math.ceil(width * factor), math.ceil(height * factor)
    while tw * th < target_mp * 1e6:  # jaga-jaga pembulatan
        tw, th = tw + 1, math.ceil((tw + 1) * height / width)
    passes = max(1, math.ceil(math.log(factor, 4) - 1e-9))
    if passes > MAX_PASSES:
        raise PlanError(
            f"Gambar terlalu kecil ({width}×{height}). Butuh perbesaran {factor:.0f}×, "
            "hasilnya tidak akan layak untuk Adobe. Generate ulang dengan resolusi lebih besar."
        )
    return Plan(width, height, tw, th, passes)
