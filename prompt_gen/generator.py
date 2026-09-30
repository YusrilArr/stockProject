"""Modul 1: ubah tema / kata kunci / ide menjadi prompt gambar yang detail."""
from __future__ import annotations

from dataclasses import dataclass

from core.ai_client import AIError, generate_text, parse_json

CONTENT_TYPES = {
    "Foto realistis": "photorealistic stock photograph; specify camera, lens, lighting, depth of field",
    "Ilustrasi digital": "digital illustration; specify art style, rendering technique, color palette",
    "Flat / vektor style": "flat vector-style illustration; clean shapes, limited palette, no gradients noise",
    "Icon": "single app/UI icon or icon set; centered, simple, consistent stroke/fill, isolated",
    "Objek terisolasi (PNG)": "single subject isolated on a plain background, ready for background removal",
    "3D render": "3D render; specify materials, lighting setup, render style",
    "Pattern / background": "seamless pattern or abstract background texture",
}

ASPECT_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]

SYSTEM_PROMPT = """You are an expert prompt engineer for AI image generators \
(Midjourney, DALL-E, Stable Diffusion, Flux, Firefly, Imagen). You write prompts \
that produce commercially usable stock imagery.

Rules for every prompt you write:
- Write in English, as one flowing descriptive paragraph (no bullet points, no model-specific \
flags like --ar or weights), so it works in any generator.
- Cover: main subject, action/pose, setting/background, composition & camera angle, \
lighting, color palette, mood, style/medium, and quality/detail descriptors.
- Be concrete and visual; avoid vague words like "beautiful" without specifics.
- Keep each prompt between 60 and 120 words.
- Each variant must take a clearly different creative angle (composition, setting, or concept), \
not just reworded copies.
- Also give a short negative prompt (comma-separated things to avoid).
"""

STOCK_RULES = """
Stock-safety rules (the image will be sold on Adobe Stock):
- No brand names, logos, trademarks, recognizable products, or text/lettering in the image.
- No real or famous people, no named artists or "in the style of <artist>", no copyrighted characters.
- Prefer compositions with useful copy space where it fits the concept.
- People (if any) should be generic, diverse, and fictional.
"""


@dataclass
class PromptResult:
    title: str
    prompt: str
    negative_prompt: str


def build_request(
    idea: str,
    content_type: str,
    aspect_ratio: str,
    n_variants: int,
    mood: str = "",
    extra: str = "",
    stock_safe: bool = True,
) -> str:
    parts = [
        f"User idea / theme / keywords: {idea.strip()}",
        f"Content type: {content_type} — {CONTENT_TYPES.get(content_type, '')}",
        f"Intended aspect ratio: {aspect_ratio} (describe a composition that suits it; "
        "do not write the ratio as a flag).",
        f"Number of variants: {n_variants}",
    ]
    if mood.strip():
        parts.append(f"Desired mood/color: {mood.strip()}")
    if extra.strip():
        parts.append(f"Extra requirements from user: {extra.strip()}")
    if stock_safe:
        parts.append(STOCK_RULES)
    parts.append(
        'Return ONLY JSON in this shape: {"variants": [{"title": "short 3-6 word label", '
        '"prompt": "...", "negative_prompt": "..."}]}'
    )
    return "\n".join(parts)


def parse_variants(text: str) -> list[PromptResult]:
    data = parse_json(text)
    items = data.get("variants", []) if isinstance(data, dict) else data
    results = []
    for it in items or []:
        if not isinstance(it, dict) or not str(it.get("prompt", "")).strip():
            continue
        results.append(
            PromptResult(
                title=str(it.get("title", "")).strip() or f"Variasi {len(results) + 1}",
                prompt=str(it["prompt"]).strip(),
                negative_prompt=str(it.get("negative_prompt", "")).strip(),
            )
        )
    if not results:
        raise AIError("AI tidak mengembalikan prompt. Coba generate ulang.")
    return results


def generate_prompts(idea: str, *, api_key: str, model: str, **opts) -> list[PromptResult]:
    if not idea.strip():
        raise AIError("Isi dulu tema, kata kunci, atau ide.")
    request = build_request(idea, **opts)
    text = generate_text(
        request, api_key=api_key, model=model, system=SYSTEM_PROMPT,
        temperature=1.0, json_output=True,
    )
    return parse_variants(text)


def to_text(results: list[PromptResult], idea: str) -> str:
    lines = [f"Ide: {idea}", ""]
    for i, r in enumerate(results, 1):
        lines += [f"=== {i}. {r.title} ===", r.prompt, ""]
        if r.negative_prompt:
            lines += [f"Negative prompt: {r.negative_prompt}", ""]
    return "\n".join(lines)
