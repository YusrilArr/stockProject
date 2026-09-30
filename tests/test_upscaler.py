import io
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image, ImageCms, ImageDraw
from streamlit.testing.v1 import AppTest

from upscaler import engine
from upscaler.detect import ICON, PHOTO, detect
from upscaler.icon import autocrop, bleed_colors
from upscaler.pipeline import Result, encode, process, to_srgb, unique_names
from upscaler.plan import PlanError, make_plan


def photo_img(size=256):
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, size)
    base = np.stack([np.outer(x, 1 - x), np.outer(1 - x, x), np.outer(x, x)], -1) * 200
    return Image.fromarray((base + rng.normal(0, 18, (size, size, 3))).clip(0, 255).astype(np.uint8))


def icon_img(size=256, margin=60):
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((margin, margin, size - margin, size - margin), fill=(230, 80, 40, 255))
    d.rectangle((size // 2 - 10, margin + 20, size // 2 + 10, size - margin - 20), fill=(255, 255, 255, 255))
    return img


def to_bytes(img, fmt="PNG"):
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


def fake_upscale(arr, key, *, threads, tile=None, progress=None):
    """Pengganti model AI di tes: Lanczos 4× (cepat, tanpa torch)."""
    if progress:
        progress(1.0)
    h, w = arr.shape[:2]
    return np.asarray(Image.fromarray(arr).resize((w * 4, h * 4), Image.LANCZOS))


# ---------- deteksi ----------

def test_detect_photo_and_icon():
    assert detect(photo_img()).kind == PHOTO
    assert detect(icon_img()).kind == ICON


def test_detect_flat_icon_without_alpha():
    flat = icon_img().convert("RGB")  # icon di background hitam solid, tanpa transparansi
    d = detect(flat)
    assert d.kind == ICON and d.transparent == 0


# ---------- rencana ukuran ----------

def test_plan_1mp_needs_2x():
    p = make_plan(1024, 1024)
    assert p.passes == 1 and p.target_w * p.target_h >= 4.2e6 and 2.0 <= p.factor < 2.1


def test_plan_keeps_aspect_and_skips_large():
    p = make_plan(1344, 768)
    assert p.target_w * p.target_h >= 4.2e6
    assert abs(p.target_w / p.target_h - 1344 / 768) < 0.01
    assert not make_plan(2048, 2048).needs_upscale


def test_plan_two_passes_and_too_small():
    assert make_plan(300, 300).passes == 2
    with pytest.raises(PlanError):
        make_plan(100, 100)


# ---------- icon ----------

def test_autocrop_and_bleed():
    img = icon_img(256, margin=60)
    cropped = autocrop(img)
    assert cropped.width < 160 and cropped.height < 160
    rgb = bleed_colors(np.asarray(img))
    # piksel transparan di pojok sekarang berwarna dekat oranye objek, bukan hitam
    assert rgb[0, 0].tolist() != [0, 0, 0] and rgb[0, 0, 0] > 150


# ---------- warna & file ----------

def test_to_srgb_profiles():
    img = photo_img(64)
    img.info["icc_profile"] = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    out, note = to_srgb(img)
    assert out.mode == "RGB" and note is None

    broken = photo_img(64)
    broken.info["icc_profile"] = b"bukan profil"
    out, note = to_srgb(broken)
    assert out.mode == "RGB" and "tidak terbaca" in note

    out, _ = to_srgb(icon_img(64, margin=10))
    assert out.mode == "RGBA"


def test_encode_embeds_srgb():
    data, ext = encode(photo_img(64), PHOTO, [])
    assert ext == ".jpg"
    prof = ImageCms.ImageCmsProfile(io.BytesIO(Image.open(io.BytesIO(data)).info["icc_profile"]))
    assert "sRGB" in ImageCms.getProfileDescription(prof)


def test_unique_names():
    rs = [Result(n, PHOTO, (1, 1), (1, 1), "", 0, b"") for n in ["a.jpg", "a.jpg", "a-2.jpg", "A.jpg"]]
    unique_names(rs)
    names = [r.filename.lower() for r in rs]
    assert len(set(names)) == 4


# ---------- alur lengkap (model AI disimulasikan) ----------

@patch("upscaler.engine.upscale_rgb", side_effect=fake_upscale)
def test_process_photo(_):
    r = process(to_bytes(photo_img(1024)), "foto sawah.png", PHOTO)
    out = Image.open(io.BytesIO(r.data))
    assert r.filename == "foto-sawah.jpg" and out.format == "JPEG"
    assert out.width * out.height >= 4.2e6 and r.engine.startswith("AI")


@patch("upscaler.engine.upscale_rgb", side_effect=fake_upscale)
def test_process_icon_keeps_transparency(_):
    r = process(to_bytes(icon_img(1024, margin=200)), "icon.png", ICON)
    out = Image.open(io.BytesIO(r.data))
    assert out.format == "PNG" and out.mode == "RGBA"
    assert out.width * out.height >= 4.2e6
    alpha = np.asarray(out.getchannel("A"))
    assert alpha[0, 0] == 0 and alpha[alpha.shape[0] // 2, alpha.shape[1] // 2] == 255
    assert any("dipangkas" in w for w in r.warnings)


@patch("upscaler.engine.upscale_rgb", side_effect=engine.EngineError("tes"))
def test_process_falls_back_to_lanczos(_):
    r = process(to_bytes(photo_img(1024)), "x.png", PHOTO)
    assert r.engine.startswith("Lanczos") and any("gagal" in w for w in r.warnings)
    assert r.out_size[0] * r.out_size[1] >= 4.2e6


def test_process_skips_large_image():
    r = process(to_bytes(photo_img(2100)), "besar.png", PHOTO)
    assert r.out_size == (2100, 2100) and r.engine.startswith("Tidak perlu")


# ---------- model AI sungguhan (dilewati bila model belum diunduh) ----------

@pytest.mark.skipif(not engine.model_path("icon_fast").exists(), reason="model belum diunduh")
def test_real_model_small_image():
    arr = np.asarray(photo_img(64))
    out = engine.upscale_rgb(arr, "icon_fast", threads=2, tile=32)
    assert out.shape == (256, 256, 3) and out.dtype == np.uint8


# ---------- halaman ----------

def test_upscaler_page_renders():
    at = AppTest.from_file("../views/upscaler.py").run()
    assert not at.exception
