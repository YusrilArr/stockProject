import streamlit as st

from core.ui import sidebar_settings

sidebar_settings()

st.title("🖼️ stockProject")
st.caption("Alat bantu kreator stock: dari ide sampai file siap upload ke Adobe Stock.")

st.markdown(
    """
**Alur kerja**

1. **✍️ Prompt Generator** — masukkan tema/ide, dapatkan prompt detail untuk generate gambar di AI mana pun.
2. **🔍 Upscaler** — perbesar gambar agar memenuhi resolusi minimal Adobe Stock (foto & icon dibedakan), diproses di komputer ini tanpa GPU.
3. **🏷️ Auto Metadata** — judul, deskripsi, keyword, kategori, rename, dan CSV siap impor ke Adobe Stock.

Setiap modul bisa dipakai sendiri-sendiri.
"""
)

st.info(
    "Pertama kali? Buat API key gratis di **aistudio.google.com → Get API key**, "
    "lalu tempel di sidebar kiri (atau simpan di file `.env`)."
)
