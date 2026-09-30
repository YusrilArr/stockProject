# stockProject

Alat bantu kreator stock (Adobe Stock): **prompt generator → upscaler → auto metadata**.
Dibuat dengan Python + Streamlit, memakai Gemini API (gratis).

| Modul | Status |
|---|---|
| 1. Prompt Generator | ✅ Siap |
| 2. Upscaler (foto vs icon, AI di CPU) | ✅ Siap |
| 3. Auto Metadata (judul, deskripsi, keyword, kategori, rename, CSV) | ✅ Siap |

## Cara menjalankan

**Persiapan (sekali saja)**
1. Install Python 3.10 atau lebih baru dari python.org.
   Di Windows, centang **"Add Python to PATH"** saat install. Install juga Git dari git-scm.com.
2. Clone repo ini:
   ```
   git clone https://github.com/YusrilArr/stockProject.git
   cd stockProject
   ```
3. Buat API key gratis: buka https://aistudio.google.com → **Get API key**.
4. Salin file `.env.example` menjadi `.env`, lalu isi `GEMINI_API_KEY=...`
   (atau tempel API key langsung di sidebar aplikasi).

**Instal manual** (kalau script di bawah gagal)
```
python -m venv .venv
.venv\Scripts\activate          # Mac: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```
Kalau instalasi terputus di tengah jalan, hapus folder `.venv` lalu ulangi.

**Menjalankan**
- Windows: klik dua kali `run_windows.bat`
- MacBook: buka Terminal di folder ini, lalu jalankan `./run_mac.sh`

Saat pertama kali dijalankan, script akan membuat virtual environment dan menginstall
library (butuh internet, ±1GB karena PyTorch, beberapa menit). Setelah itu browser akan terbuka di `http://localhost:8501`.

## Catatan model
- Default: `gemini-3.5-flash-lite` (kuota gratis harian paling besar).
- Kalau muncul error "model tidak ditemukan" atau "kuota habis", ganti model di sidebar.
  Daftar model & kuota gratis Google bisa berubah; cek di aistudio.google.com.
- Data di free tier Gemini bisa dipakai Google untuk meningkatkan produknya.
  Jangan kirim data sensitif.

## Catatan upscaler
- Berjalan di CPU (tanpa GPU), memakai Real-ESRGAN lewat PyTorch.
- File model disimpan di folder `models/` dan diunduh otomatis dari GitHub saat pertama dipakai
  (mode Cepat ±7MB, mode Maksimal +85MB).
- MacBook Intel: memakai PyTorch 2.2.2 (versi terakhir untuk Mac Intel) dan butuh Python 3.10 atau lebih baru.

## Struktur
```
app.py              # entry point & menu
core/               # config, client Gemini, sidebar (dipakai semua modul)
prompt_gen/         # logika Modul 1
upscaler/           # logika Modul 2
metadata/           # logika Modul 3
views/              # halaman Streamlit
tests/              # jalankan: python -m pytest tests
```
