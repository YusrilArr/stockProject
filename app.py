"""stockProject — entry point. Jalankan: streamlit run app.py"""
import streamlit as st

st.set_page_config(page_title="stockProject", page_icon="🖼️", layout="wide")

home = st.Page("views/home.py", title="Beranda", icon="🏠", default=True)
prompt = st.Page("views/prompt_generator.py", title="1. Prompt Generator", icon="✍️")
upscale = st.Page("views/upscaler.py", title="2. Upscaler", icon="🔍")
metadata = st.Page("views/metadata.py", title="3. Auto Metadata", icon="🏷️")

st.navigation([home, prompt, upscale, metadata]).run()
