"""Konfigurasi global aplikasi."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

# Model Gemini yang bisa dipilih. Flash-Lite punya kuota gratis harian paling besar.
# Daftar model & kuota bisa berubah: cek https://aistudio.google.com
MODEL_OPTIONS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", MODEL_OPTIONS[0]).strip()


def env_api_key() -> str:
    return os.getenv("GEMINI_API_KEY", "").strip()
