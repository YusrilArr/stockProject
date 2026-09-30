import streamlit as st

from core.ai_client import AIError
from core.ui import sidebar_settings
from prompt_gen.generator import ASPECT_RATIOS, CONTENT_TYPES, generate_prompts, to_text

api_key, model = sidebar_settings()

st.title("✍️ Prompt Generator")
st.caption("Masukkan tema, kata kunci, atau ide. Hasilnya prompt detail (bahasa Inggris) "
           "yang bisa dipakai di Midjourney, DALL·E, Stable Diffusion, Firefly, dan lainnya.")

with st.form("prompt_form"):
    idea = st.text_area(
        "Tema / kata kunci / ide",
        placeholder="Contoh: suasana pagi di sawah terasering Bali, petani, kabut tipis",
        height=100,
    )
    c1, c2, c3 = st.columns(3)
    content_type = c1.selectbox("Jenis konten", list(CONTENT_TYPES))
    aspect_ratio = c2.selectbox("Rasio gambar", ASPECT_RATIOS)
    n_variants = c3.slider("Jumlah variasi", 1, 5, 3)

    with st.expander("Opsi tambahan"):
        mood = st.text_input("Mood / warna (opsional)", placeholder="hangat, golden hour, pastel")
        extra = st.text_input("Syarat lain (opsional)", placeholder="ada ruang kosong di kiri untuk teks")
        stock_safe = st.checkbox(
            "Aman untuk stock (tanpa logo, merek, teks, orang terkenal, nama seniman)",
            value=True,
        )
    submitted = st.form_submit_button("Generate prompt", type="primary", use_container_width=True)

if submitted:
    try:
        with st.spinner("Membuat prompt..."):
            results = generate_prompts(
                idea, api_key=api_key, model=model,
                content_type=content_type, aspect_ratio=aspect_ratio,
                n_variants=n_variants, mood=mood, extra=extra, stock_safe=stock_safe,
            )
        st.session_state["prompt_results"] = results
        st.session_state["prompt_idea"] = idea.strip()
    except AIError as e:
        st.error(str(e))

results = st.session_state.get("prompt_results")
if results:
    st.divider()
    head, dl = st.columns([3, 1])
    head.subheader(f"Hasil ({len(results)} variasi)")
    dl.download_button(
        "⬇️ Download .txt",
        to_text(results, st.session_state.get("prompt_idea", "")),
        file_name="prompts.txt",
        use_container_width=True,
    )
    st.caption("Klik ikon salin di pojok kanan atas tiap kotak untuk menyalin prompt.")
    for i, r in enumerate(results, 1):
        st.markdown(f"**{i}. {r.title}**")
        st.code(r.prompt, language=None, wrap_lines=True)
        if r.negative_prompt:
            with st.expander("Negative prompt (untuk Stable Diffusion / Flux)"):
                st.code(r.negative_prompt, language=None, wrap_lines=True)
