"""Client Gemini yang dipakai bersama oleh semua modul."""
from __future__ import annotations

import json
import re
import time

from google import genai
from google.genai import types


class AIError(Exception):
    """Error yang pesannya aman ditampilkan ke user."""


def _client(api_key: str) -> genai.Client:
    if not api_key:
        raise AIError("API key Gemini belum diisi. Isi di sidebar atau di file .env.")
    return genai.Client(api_key=api_key)


def generate_text(
    prompt: str,
    *,
    api_key: str,
    model: str,
    system: str | None = None,
    images: list[tuple[bytes, str]] | None = None,
    temperature: float = 0.9,
    json_output: bool = False,
    max_retries: int = 3,
) -> str:
    """Kirim prompt (opsional dengan gambar) dan kembalikan teks jawaban.

    images: list (bytes, mime_type) untuk modul yang butuh vision (metadata).
    """
    client = _client(api_key)
    contents: list = []
    for data, mime in images or []:
        contents.append(types.Part.from_bytes(data=data, mime_type=mime))
    contents.append(prompt)

    config = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        response_mime_type="application/json" if json_output else None,
        # Tidak memakai function calling; dimatikan agar tidak muncul peringatan AFC di terminal.
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
            text = (resp.text or "").strip()
            if not text:
                raise AIError("AI mengembalikan jawaban kosong. Coba lagi.")
            return text
        except AIError:
            raise
        except Exception as e:  # noqa: BLE001
            last_err = e
            msg = str(e)
            if "API_KEY_INVALID" in msg or "API key not valid" in msg:
                raise AIError("API key tidak valid. Cek lagi di Google AI Studio.") from e
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise AIError(
                    "Kuota gratis Gemini habis atau terlalu banyak request per menit. "
                    "Tunggu sebentar, atau ganti model di sidebar."
                ) from e
            if "404" in msg or "NOT_FOUND" in msg:
                raise AIError(f"Model '{model}' tidak ditemukan. Pilih model lain di sidebar.") from e
            if attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
    raise AIError(f"Gagal menghubungi Gemini: {last_err}")


def parse_json(text: str):
    """Parse JSON dari jawaban AI, toleran terhadap pembungkus ```json ... ```."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        raise AIError("Format jawaban AI tidak terbaca. Coba generate ulang.")
