"""Komponen UI yang dipakai bersama semua halaman."""
import streamlit as st

from core import config


def sidebar_settings() -> tuple[str, str]:
    """Tampilkan pengaturan API di sidebar. Return (api_key, model)."""
    with st.sidebar:
        st.markdown("### ⚙️ Pengaturan AI")
        default_key = st.session_state.get("api_key") or config.env_api_key()
        api_key = st.text_input(
            "Gemini API key",
            value=default_key,
            type="password",
            help="Gratis dari https://aistudio.google.com → Get API key. "
            "Bisa juga disimpan di file .env agar tidak perlu diisi ulang.",
        )
        st.session_state["api_key"] = api_key.strip()

        default_model = st.session_state.get("model", config.DEFAULT_MODEL)
        options = list(config.MODEL_OPTIONS)
        if default_model not in options:
            options.insert(0, default_model)
        model = st.selectbox(
            "Model",
            options,
            index=options.index(default_model),
            help="Flash-Lite: kuota gratis harian paling besar. "
            "Flash: kualitas lebih baik, kuota gratis lebih kecil.",
        )
        st.session_state["model"] = model

        if not api_key:
            st.warning("API key belum diisi.")
        else:
            st.success("API key terisi.")
    return st.session_state["api_key"], model
