"""Modul 3: Gemini membaca gambar lalu membuat judul, deskripsi, keyword, dan kategori Adobe Stock."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from PIL import Image

from core.ai_client import AIError, generate_text, parse_json

# Aturan CSV Adobe Stock (diverifikasi 29 Sep 2026, lihat docs/spec.md bagian 7).
TITLE_MAX = 70
DESCRIPTION_MAX = 200
KEYWORDS_MAX = 49
KEYWORDS_DEFAULT = 25
# Dibuang dari keyword hasil AI: menyesatkan untuk file raster, atau soal cara pembuatan
# (label AI diisi lewat centang di portal Adobe, bukan keyword).
BANNED_KEYWORDS = {"vector", "vectors", "clipart", "clip art", "svg", "eps", "ai", "ai generated",
                   "generative ai", "ai art", "generated"}

# Nomor kategori CSV Adobe Stock. Sumber: Adobe Community (2018); cocokkan di dialog upload CSV.
CATEGORIES = {
    1: "Animals", 2: "Buildings and Architecture", 3: "Business", 4: "Drinks",
    5: "The Environment", 6: "States of Mind", 7: "Food", 8: "Graphic Resources",
    9: "Hobbies and Leisure", 10: "Industry", 11: "Landscapes", 12: "Lifestyle",
    13: "People", 14: "Plants and Flowers", 15: "Culture and Religion", 16: "Science",
    17: "Social Issues", 18: "Sports", 19: "Technology", 20: "Transport", 21: "Travel",
}

SYSTEM_PROMPT = """You are an expert Adobe Stock metadata writer. You look at one image and write \
metadata that helps buyers find it in search.

Rules:
- Everything in English.
- title: a short descriptive phrase (not a formal sentence) naming the main subject and the most \
important visual context, at most 70 characters, NO commas.
- description: one or two plain sentences, at most 200 characters, describing what is visible.
- keywords: single words or short 2-word phrases, lowercase, ordered from most to least important. \
The first 10 must contain every important word from the title and the main subject. Then add \
setting, colors, mood, concepts, and likely buyer uses. No duplicates or plural/singular repeats.
- Never include brand names, logos, real or famous people, artist names, fictional character names, \
names of films/franchises/comics, or phrases like "in the style of".
- Do not describe how the image was made (no "AI", "generated", "render settings", camera settings).
- The file is a raster image (JPEG/PNG), not a vector file: never use the words "vector", "clipart", \
"svg", or "eps" anywhere; say "illustration" or "flat illustration" instead.
- Do not invent specific places, events, or dates that are not clearly visible or given by the user.
- category: the single best matching number from the category list.
"""


@dataclass
class ImageMeta:
    source: str                       # nama file asli
    filename: str                     # nama file baru (hasil rename)
    title: str
    description: str
    keywords: list[str] = field(default_factory=list)
    category: int | None = None


def category_list_text() -> str:
    return "\n".join(f"{n}: {name}" for n, name in CATEGORIES.items())


def build_request(kind: str, n_keywords: int, context: str = "") -> str:
    parts = [
        f"Image type: {kind}.",
        f"Write exactly {n_keywords} keywords.",
    ]
    if "icon" in kind.lower() or "png" in kind.lower():
        parts.append("The real file is an isolated subject on a TRANSPARENT background (the preview "
                     "is shown on white only for viewing). Describe the background as transparent, "
                     "never as white. Mention 'isolated' and 'transparent background' in the keywords "
                     "and, if it fits, 'isolated' in the title.")
    if context.strip():
        parts.append(f"Extra context from the contributor (trust it for place/subject names): {context.strip()}")
    parts.append("Adobe Stock categories:\n" + category_list_text())
    parts.append('Return ONLY JSON: {"title": "...", "description": "...", '
                 '"keywords": ["...", "..."], "category": 11}')
    return "\n".join(parts)


# ---------- pembersihan hasil AI ----------

def clean_title(text: str, limit: int = TITLE_MAX) -> str:
    """Hapus koma & spasi berlebih, potong di batas kata agar ≤ limit karakter."""
    t = re.sub(r"\s*,\s*", " ", str(text or ""))
    t = re.sub(r"\s+", " ", t).strip().strip(".;:-–— ")
    if len(t) > limit:
        cut = t[:limit + 1].rsplit(" ", 1)[0]
        t = (cut if len(cut) >= limit // 2 else t[:limit]).strip().strip(".;:-–— ")
    return t[:1].upper() + t[1:]


def clean_description(text: str, limit: int = DESCRIPTION_MAX) -> str:
    t = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(t) > limit:
        t = t[:limit + 1].rsplit(" ", 1)[0].rstrip(",;:") + "."
    return t


def clean_keywords(items, limit: int = KEYWORDS_MAX) -> list[str]:
    """Rapikan keyword: huruf kecil, tanpa duplikat, urutan dipertahankan, maks `limit`."""
    if isinstance(items, str):
        items = items.split(",")
    out, seen = [], set()
    for raw in items or []:
        for part in str(raw).split(","):
            k = re.sub(r"\s+", " ", part).strip().strip(".;:\"'#").lower()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(k)
    return out[:limit]


def clean_category(value) -> int | None:
    try:
        n = int(str(value).strip().split()[0].rstrip(".:"))
    except (ValueError, IndexError):
        return None
    return n if n in CATEGORIES else None


def parse_metadata(text: str, source: str, n_keywords: int) -> ImageMeta:
    data = parse_json(text)
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        raise AIError("Format jawaban AI tidak sesuai. Coba generate ulang.")
    title = clean_title(data.get("title", ""))
    keywords = [k for k in clean_keywords(data.get("keywords", []), limit=10_000) if k not in BANNED_KEYWORDS]
    keywords = keywords[:min(n_keywords, KEYWORDS_MAX)]
    if not title or not keywords:
        raise AIError("AI tidak mengembalikan judul atau keyword. Coba generate ulang.")
    return ImageMeta(
        source=source, filename="", title=title,
        description=clean_description(data.get("description", "")),
        keywords=keywords, category=clean_category(data.get("category")),
    )


# ---------- pemanggilan AI ----------

def preview_for_ai(data: bytes, max_side: int = 1024) -> tuple[bytes, str]:
    """Perkecil gambar sebelum dikirim ke Gemini: lebih cepat & hemat kuota, isi tetap terbaca.

    Gambar transparan ditempel di atas putih lalu dikirim sebagai JPEG.
    """
    img = Image.open(io.BytesIO(data))
    img.thumbnail((max_side, max_side))
    if img.mode in ("RGBA", "LA", "P"):
        rgba = img.convert("RGBA")
        img = Image.new("RGB", rgba.size, "white")
        img.paste(rgba, mask=rgba.getchannel("A"))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=88)
    return buf.getvalue(), "image/jpeg"


def generate_metadata(
    data: bytes,
    source: str,
    kind: str,
    *,
    api_key: str,
    model: str,
    n_keywords: int = KEYWORDS_DEFAULT,
    context: str = "",
) -> ImageMeta:
    try:
        image = preview_for_ai(data)
    except Exception as e:  # noqa: BLE001
        raise AIError(f"{source}: file bukan gambar yang bisa dibaca.") from e
    text = generate_text(
        build_request(kind, n_keywords, context),
        api_key=api_key, model=model, system=SYSTEM_PROMPT,
        images=[image], temperature=0.4, json_output=True,
    )
    return parse_metadata(text, source, n_keywords)
