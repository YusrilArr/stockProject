import base64
import io
import sys
import time

import pandas as pd
import streamlit as st
from PIL import Image

from core.ui import sidebar_settings
from upscaler import engine
from upscaler.detect import ICON, KINDS, detect
from upscaler.pipeline import MAX_BATCH, open_image, process, save_results, unique_names, zip_results
from upscaler.plan import PlanError, make_plan

sidebar_settings()

with st.sidebar:
    st.markdown("### 🔍 Pengaturan upscale")
    quality = st.radio(
        "Kualitas",
        ["fast", "best"],
        format_func={"fast": "Cepat (disarankan)", "best": "Maksimal (jauh lebih lama)"}.get,
        help="Cepat: ±5–10 detik per gambar di PC. Maksimal: model besar, bisa 1–3 menit per gambar.",
    )
    save_power = st.toggle(
        "Mode hemat CPU",
        value=sys.platform == "darwin",
        help="Memakai separuh core CPU. Proses sedikit lebih lama, tapi komputer tetap nyaman "
        "dipakai dan laptop tidak terlalu panas.",
    )
threads = engine.default_threads(save_power)

st.title("🔍 Upscaler")
st.caption("Perbesar gambar AI sampai memenuhi syarat Adobe Stock (minimal 4MP). "
           "Foto/ilustrasi disimpan sebagai JPEG, icon sebagai PNG transparan. Semua diproses di komputer ini.")

ok, msg = engine.engine_available()
if not ok:
    st.warning(f"{msg}\n\nSementara ini upscale memakai resize Lanczos (kualitas lebih rendah).")


def _thumb(img: Image.Image, size: int = 96) -> str:
    t = img.copy()
    t.thumbnail((size, size))
    buf = io.BytesIO()
    t.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _analyze(f) -> dict:
    cache = st.session_state.setdefault("up_info", {})
    if f.file_id not in cache:
        try:
            img = open_image(f.getvalue())
            d = detect(img)
            info = {"img": img, "kind": d.kind, "reason": d.reason, "error": None}
        except ValueError as e:
            info = {"img": None, "kind": KINDS[0], "reason": "", "error": str(e)}
        cache[f.file_id] = info
    return cache[f.file_id]


def _plan_text(img: Image.Image, kind: str) -> str:
    try:
        p = make_plan(*img.size)
    except PlanError as e:
        return f"⚠️ {e}"
    if not p.needs_upscale:
        return "Sudah ≥4MP, tidak di-upscale"
    txt = f"{p.target_w}×{p.target_h} ({p.target_mp:.1f}MP, {p.factor:.1f}×)".replace(".", ",")
    return txt + (" · dihitung ulang setelah dipangkas" if kind == ICON else "")


files = st.file_uploader(
    f"Upload gambar (maksimal {MAX_BATCH})",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
)
if len(files) > MAX_BATCH:
    st.warning(f"Hanya {MAX_BATCH} gambar pertama yang diproses.")
    files = files[:MAX_BATCH]

if files:
    infos = [_analyze(f) for f in files]
    table = pd.DataFrame([
        {
            "Preview": _thumb(i["img"]) if i["img"] else None,
            "File": f.name,
            "Ukuran": f"{i['img'].width}×{i['img'].height}" if i["img"] else "—",
            "Jenis": i["kind"],
            "Rencana": _plan_text(i["img"], i["kind"]) if i["img"] else f"⚠️ {i['error']}",
            "Deteksi": i["reason"],
        }
        for f, i in zip(files, infos)
    ])
    st.caption("Jenis dideteksi otomatis. Klik kolom **Jenis** untuk mengoreksi bila salah.")
    edited = st.data_editor(
        table,
        hide_index=True,
        width="stretch",
        disabled=["Preview", "File", "Ukuran", "Rencana", "Deteksi"],
        column_config={
            "Preview": st.column_config.ImageColumn("Preview", width="small"),
            "Jenis": st.column_config.SelectboxColumn("Jenis", options=KINDS, required=True),
        },
        key=f"up_table_{'-'.join(f.file_id for f in files)}",
    )

    jobs = [(f, i, kind) for f, i, kind in zip(files, infos, edited["Jenis"]) if i["img"] is not None]
    if st.button(f"🚀 Upscale {len(jobs)} gambar", type="primary", width="stretch", disabled=not jobs):
        results, failed = [], []
        bar = st.progress(0.0, text="Memulai...")
        t0 = time.time()
        for n, (f, _, kind) in enumerate(jobs):
            def on_progress(frac, text, n=n, name=f.name):
                bar.progress((n + frac) / len(jobs), text=f"{n + 1}/{len(jobs)} · {name} · {text}")
            try:
                results.append(process(f.getvalue(), f.name, kind, quality=quality,
                                       threads=threads, progress=on_progress))
            except (ValueError, PlanError, engine.EngineError) as e:
                failed.append(f"{f.name}: {e}")
        bar.empty()
        unique_names(results)
        st.session_state["upscaled"] = results
        st.session_state["upscale_sources"] = {f.name: i["img"] for f, i, _ in jobs}
        st.session_state["upscale_folder"] = str(save_results(results)) if results else None
        st.session_state["upscale_failed"] = failed
        st.session_state["upscale_seconds"] = time.time() - t0


def _zoom_pair(src: Image.Image, out: Image.Image, box: int = 320) -> tuple[Image.Image, Image.Image]:
    """Potongan tengah pada zoom 100%: resize biasa vs hasil upscale, dari area yang sama."""
    cx, cy = out.width // 2, out.height // 2
    region = (cx - box // 2, cy - box // 2, cx + box // 2, cy + box // 2)
    plain = src.convert(out.mode).resize(out.size, Image.LANCZOS) if src.size != out.size else src
    return plain.crop(region), out.crop(region)


results = st.session_state.get("upscaled")
if results is not None:
    st.divider()
    for err in st.session_state.get("upscale_failed", []):
        st.error(err)
    if results:
        secs = st.session_state.get("upscale_seconds", 0)
        st.success(f"{len(results)} gambar selesai dalam {secs:.0f} detik. "
                   f"Disimpan di folder `{st.session_state['upscale_folder']}`.")
        st.download_button("⬇️ Download semua (ZIP)", zip_results(results),
                           file_name="upscaled.zip", mime="application/zip", width="stretch")
        sources = st.session_state.get("upscale_sources", {})
        for r in results:
            with st.container(border=True):
                out = Image.open(io.BytesIO(r.data))
                head, dl = st.columns([3, 1])
                head.markdown(f"**{r.filename}** · {r.kind}")
                head.caption(
                    f"{r.src_size[0]}×{r.src_size[1]} → {r.out_size[0]}×{r.out_size[1]} "
                    f"({r.out_size[0] * r.out_size[1] / 1e6:.1f}MP) · {r.size_mb:.1f}MB · "
                    f"{r.seconds:.1f} dtk · {r.engine}"
                )
                dl.download_button("⬇️ Download", r.data, file_name=r.filename, mime=r.mime,
                                   key=f"dl_{r.filename}", width="stretch")
                for w in r.warnings:
                    st.warning(w)
                src = sources.get(r.source)
                if src is not None and src.size != out.size:
                    with st.expander("Bandingkan zoom 100%: resize biasa vs upscale"):
                        plain, ai = _zoom_pair(src, out)
                        c1, c2 = st.columns(2)
                        c1.image(plain, caption="Resize biasa (Lanczos)", width="stretch")
                        c2.image(ai, caption="Hasil upscale", width="stretch")
                else:
                    st.image(out, width=320)
        st.info("Lanjut ke **🏷️ Auto Metadata** untuk membuat judul, deskripsi, dan keyword. "
                "Jangan kirim lebih dari ±3 gambar yang mirip per konsep ke Adobe.")
