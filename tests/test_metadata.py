import csv
import io
import json
import zipfile
from unittest.mock import patch

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from core.ai_client import AIError
from metadata.files import (CSV_HEADER, FILENAME_MAX, assign_filenames, build_csv, make_filename,
                            save_package, validate, zip_package)
from metadata.generator import (ImageMeta, build_request, clean_category, clean_keywords, clean_title,
                                generate_metadata, parse_metadata, preview_for_ai)

FAKE = json.dumps({
    "title": "Misty Bali rice terraces at sunrise, with farmer",
    "description": "Green rice terraces in morning mist with a farmer walking on a path.",
    "keywords": ["Rice Terrace", "bali", "sunrise", "rice terrace", "farmer, agriculture", "", "mist."],
    "category": "11",
})


def jpg_bytes(size=(64, 48), color=(40, 160, 60)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def meta(source="a.jpg", filename="a.jpg", title="Title", keywords=("one", "two"), category=11):
    return ImageMeta(source, filename, title, "desc", list(keywords), category)


# ---------- pembersihan ----------

def test_clean_title_removes_commas_and_limits_length():
    assert clean_title("sunset, beach , palm trees.") == "Sunset beach palm trees"
    long = "word " * 30
    t = clean_title(long)
    assert len(t) <= 70 and not t.endswith(" ")


def test_clean_keywords():
    kws = clean_keywords(["Rice Terrace", "rice terrace", "a, b", " ", "mist."])
    assert kws == ["rice terrace", "a", "b", "mist"]
    assert len(clean_keywords([f"k{i}" for i in range(80)])) == 49


def test_clean_category():
    assert clean_category("11") == 11 and clean_category("11 · Landscapes") == 11
    assert clean_category(99) is None and clean_category("x") is None


def test_parse_metadata():
    m = parse_metadata("```json\n" + FAKE + "\n```", "foto.jpg", 25)
    assert "," not in m.title and m.category == 11
    assert m.keywords[:2] == ["rice terrace", "bali"] and "agriculture" in m.keywords
    with pytest.raises(AIError):
        parse_metadata('{"title": "", "keywords": []}', "x.jpg", 25)


def test_parse_metadata_drops_misleading_keywords_and_limits():
    raw = json.dumps({"title": "Apple", "keywords": ["apple", "Vector", "clipart", "AI generated"]
                      + [f"k{i}" for i in range(40)]})
    m = parse_metadata(raw, "a.png", 20)
    assert m.keywords[0] == "apple" and "vector" not in m.keywords and len(m.keywords) == 20


def test_build_request_icon_and_context():
    r = build_request("Icon / PNG transparan", 30, "Bali")
    assert "exactly 30 keywords" in r and "isolated" in r and "Bali" in r and "11: Landscapes" in r


def test_preview_for_ai_flattens_transparency():
    buf = io.BytesIO()
    Image.new("RGBA", (2000, 1000), (0, 0, 0, 0)).save(buf, "PNG")
    data, mime = preview_for_ai(buf.getvalue())
    img = Image.open(io.BytesIO(data))
    assert mime == "image/jpeg" and max(img.size) == 1024 and img.getpixel((0, 0)) == (255, 255, 255)


def test_generate_metadata_sends_image():
    with patch("metadata.generator.generate_text", return_value=FAKE) as m:
        r = generate_metadata(jpg_bytes(), "foto.jpg", "Foto / ilustrasi", api_key="k", model="m")
    assert r.title.startswith("Misty") and m.call_args.kwargs["images"][0][1] == "image/jpeg"


# ---------- nama file ----------

def test_make_filename_limits_and_unique():
    used = set()
    a = make_filename("Misty Bali rice terraces at sunrise with farmer", ".jpg", used)
    b = make_filename("Misty Bali rice terraces at sunrise with farmer", ".jpg", used)
    assert len(a) <= FILENAME_MAX and len(b) <= FILENAME_MAX and a != b
    assert a.endswith(".jpg") and a.startswith("misty-bali-rice")
    assert make_filename("Café ñandú!!", "PNG", set()) == "cafe-nandu.png"
    assert make_filename("", ".jpg", set()) == "image.jpg"


def test_assign_filenames_keeps_extension():
    items = [meta("x.png", "", "Red apple"), meta("y.jpg", "", "Red apple")]
    assign_filenames(items)
    assert items[0].filename == "red-apple.png" and items[1].filename == "red-apple.jpg"


# ---------- validasi & ekspor ----------

def test_validate():
    ok = [meta()]
    assert validate(ok) == ([], [])
    bad = [meta(filename="a" * 40 + ".jpg", title="a, b", keywords=[], category=None),
           meta(source="b.png", filename="b.jpg")]
    errors, warnings = validate(bad)
    text = " ".join(errors)
    assert "maksimal 30" in text and "koma" in text and "keyword kosong" in text and "ekstensi" in text
    assert any("kategori" in w for w in warnings)


def test_build_csv_matches_adobe_format():
    s = build_csv([meta(keywords=["rice terrace", "bali"], category=11), meta("b.jpg", "b.jpg", category=None)])
    assert s.startswith("Filename,Title,Keywords,Category,Releases\n") and "\r" not in s
    rows = list(csv.reader(io.StringIO(s)))
    assert rows[0] == CSV_HEADER
    assert rows[1] == ["a.jpg", "Title", "rice terrace, bali", "11", ""]
    assert rows[2][3] == ""


def test_save_and_zip_package(tmp_path):
    items = [meta("asli.jpg", "baru.jpg")]
    files = {"asli.jpg": jpg_bytes()}
    folder = save_package(items, files, root=tmp_path)
    assert (folder / "baru.jpg").read_bytes() == files["asli.jpg"]
    assert (folder / "adobe_stock_metadata.csv").read_bytes().startswith(b"Filename,")
    z = zipfile.ZipFile(io.BytesIO(zip_package(items, files)))
    assert sorted(z.namelist()) == ["adobe_stock_metadata.csv", "baru.jpg"]


# ---------- halaman ----------

def test_metadata_page_renders():
    at = AppTest.from_file("../views/metadata.py").run()
    assert not at.exception


def test_metadata_page_with_upscaled_results():
    from upscaler.pipeline import Result
    r = Result("sawah.jpg", "Foto / ilustrasi", (64, 48), (64, 48), "", 0, jpg_bytes(), source="sawah.png")
    with patch("metadata.generator.generate_text", return_value=FAKE):
        at = AppTest.from_file("../views/metadata.py")
        at.session_state["upscaled"] = [r]
        at.run()
        at.sidebar.text_input[0].input("dummy-key")
        at.button[0].click().run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    assert any("siap diekspor" in s.value for s in at.success)
