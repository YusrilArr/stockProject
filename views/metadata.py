import base64
import io
import time

import pandas as pd
import streamlit as st
from PIL import Image

from core.ai_client import AIError
from core.ui import sidebar_settings
from metadata.files import (CSV_NAME, FILENAME_MAX, assign_filenames, build_csv, save_package,
                            validate, zip_package)
from metadata.generator import (CATEGORIES, KEYWORDS_DEFAULT, KEYWORDS_MAX, TITLE_MAX, ImageMeta,
                                clean_category, clean_keywords, generate_metadata)
from upscaler.detect import detect
from upscaler.pipeline import MAX_BATCH

api_key, model = sidebar_settings()

st.title("🏷️ Auto Metadata")
st.caption("AI membaca gambar lalu membuat judul, deskripsi, keyword, dan kategori. Periksa dan edit "
           "di tabel, lalu ekspor file yang sudah di-rename beserta CSV siap impor ke Adobe Stock.")

CAT_OPTIONS = [f"{n} · {name}" for n, name in CATEGORIES.items()]


def _cat_label(n: int | None) -> str | None:
    return f"{n} · {CATEGORIES[n]}" if n in CATEGORIES else None


def _thumb(data: bytes, size: int = 96) -> str:
    img = Image.open(io.BytesIO(data))
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ---------- 1. sumber gambar ----------

upscaled = st.session_state.get("upscaled") or []
use_upscaled = False
if upscaled:
    choice = st.radio(
        "Sumber gambar",
        [f"Hasil Upscaler ({len(upscaled)} gambar)", "Upload gambar baru"],
        horizontal=True,
    )
    use_upscaled = choice.startswith("Hasil")

sources: list[tuple[str, bytes, str]] = []   # (nama unik, data, jenis)
if use_upscaled:
    sources = [(r.filename, r.data, r.kind) for r in upscaled]
else:
    files = st.file_uploader(f"Upload gambar (maksimal {MAX_BATCH})", type=["png", "jpg", "jpeg"],
                             accept_multiple_files=True)
    if len(files) > MAX_BATCH:
        st.warning(f"Hanya {MAX_BATCH} gambar pertama yang diproses.")
        files = files[:MAX_BATCH]
    names: set[str] = set()
    for f in files:
        name, n = f.name, 1
        while name.lower() in names:  # nama upload kembar → beri akhiran
            n += 1
            stem, _, ext = f.name.rpartition(".")
            name = f"{stem}-{n}.{ext}"
        names.add(name.lower())
        data = f.getvalue()
        try:
            kind = detect(Image.open(io.BytesIO(data))).kind
        except Exception:  # noqa: BLE001
            st.error(f"{f.name}: file bukan gambar yang bisa dibaca.")
            continue
        sources.append((name, data, kind))

if sources:
    cols = st.columns(min(len(sources), 8))
    for c, (name, data, _) in zip(cols, sources):
        c.image(data, caption=name, width="stretch")

# ---------- 2. generate ----------

c1, c2 = st.columns([1, 2])
n_keywords = c1.slider("Jumlah keyword", 15, KEYWORDS_MAX, KEYWORDS_DEFAULT,
                       help="Adobe menyarankan 15–25 keyword yang relevan. Maksimal 49.")
context = c2.text_input(
    "Konteks tambahan (opsional)",
    placeholder="Contoh: lokasi Bali, Indonesia · untuk tema Ramadan",
    help="AI tidak tahu lokasi atau tujuan gambar. Isi di sini bila perlu. Berlaku untuk semua gambar.",
)

if st.button(f"✨ Buat metadata untuk {len(sources)} gambar", type="primary", width="stretch",
             disabled=not sources):
    if not api_key:
        st.error("API key Gemini belum diisi di sidebar.")
    else:
        items, failed = [], []
        bar = st.progress(0.0, text="Memulai...")
        t0 = time.time()
        for i, (name, data, kind) in enumerate(sources):
            bar.progress(i / len(sources), text=f"{i + 1}/{len(sources)} · {name}")
            try:
                items.append(generate_metadata(data, name, kind, api_key=api_key, model=model,
                                               n_keywords=n_keywords, context=context))
            except AIError as e:
                failed.append(f"{name}: {e}")
        bar.empty()
        assign_filenames(items)
        st.session_state["meta_items"] = items
        st.session_state["meta_files"] = {name: data for name, data, _ in sources}
        st.session_state["meta_failed"] = failed
        st.session_state["meta_seconds"] = time.time() - t0
        st.session_state["meta_version"] = st.session_state.get("meta_version", 0) + 1

# ---------- 3. periksa & edit ----------

items: list[ImageMeta] = st.session_state.get("meta_items") or []
files_map: dict[str, bytes] = st.session_state.get("meta_files") or {}
for err in st.session_state.get("meta_failed", []):
    st.error(err)

if items:
    st.divider()
    st.subheader(f"Periksa & edit ({len(items)} gambar)")
    st.caption(f"Judul maks {TITLE_MAX} karakter tanpa koma · nama file maks {FILENAME_MAX} karakter · "
               "keyword dipisah koma, yang terpenting di depan (10 pertama paling berpengaruh).")
    table = pd.DataFrame([
        {
            "Preview": _thumb(files_map[it.source]) if it.source in files_map else None,
            "Nama file": it.filename,
            "Judul": it.title,
            "Keyword": ", ".join(it.keywords),
            "Kategori": _cat_label(it.category),
            "Deskripsi": it.description,
            "Asal": it.source,
        }
        for it in items
    ])
    edited = st.data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=["Preview", "Asal"],
        column_config={
            "Preview": st.column_config.ImageColumn("Preview", width="small"),
            "Nama file": st.column_config.TextColumn("Nama file", max_chars=FILENAME_MAX),
            "Judul": st.column_config.TextColumn("Judul", width="medium"),
            "Keyword": st.column_config.TextColumn("Keyword", width="large"),
            "Kategori": st.column_config.SelectboxColumn("Kategori", options=CAT_OPTIONS),
            "Deskripsi": st.column_config.TextColumn("Deskripsi", width="medium"),
            "Asal": st.column_config.TextColumn("File asal", width="small"),
        },
        key=f"meta_table_{st.session_state.get('meta_version', 0)}",
    )

    current = [
        ImageMeta(
            source=row["Asal"],
            filename=str(row["Nama file"] or "").strip(),
            title=str(row["Judul"] or "").strip(),
            description=str(row["Deskripsi"] or "").strip(),
            keywords=clean_keywords(row["Keyword"] or "", limit=10_000),
            category=clean_category(row["Kategori"]) if row["Kategori"] else None,
        )
        for _, row in edited.iterrows()
    ]

    if st.button("↻ Buat ulang nama file dari judul",
                 help="Pakai setelah mengedit judul, supaya nama file ikut menyesuaikan."):
        assign_filenames(current)
        st.session_state["meta_items"] = current
        st.session_state["meta_version"] = st.session_state.get("meta_version", 0) + 1
        st.rerun()

    st.caption(" · ".join(f"**{it.filename}**: {len(it.title)} karakter, {len(it.keywords)} keyword"
                          for it in current))
    errors, warnings = validate(current)
    for e in errors:
        st.error(e)
    for w in warnings:
        st.warning(w)
    if not errors:
        st.success("Metadata siap diekspor.")

    # ---------- 4. ekspor ----------
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇️ Download CSV", build_csv(current), file_name=CSV_NAME, mime="text/csv",
                       width="stretch", disabled=bool(errors))
    d2.download_button("⬇️ Download ZIP (file + CSV)",
                       zip_package(current, files_map) if not errors else b"",
                       file_name="adobe_stock_upload.zip", mime="application/zip",
                       width="stretch", disabled=bool(errors))
    if d3.button("💾 Simpan ke folder output", width="stretch", disabled=bool(errors)):
        folder = save_package(current, files_map)
        st.success(f"File yang sudah di-rename dan `{CSV_NAME}` disimpan di `{folder}`.")

    st.info(
        "**Setelah upload ke Adobe Stock:** upload file gambarnya, lalu impor CSV di Contributor Portal. "
        "Setelah itu centang **Created using generative AI tools** secara manual, karena label ini tidak "
        "bisa diisi lewat CSV. Kalau ada orang atau properti yang tampak nyata, centang juga "
        "**People and Property are fictional**. Kirim maksimal ±3 gambar mirip per konsep."
    )
