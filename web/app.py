import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import requests
import cv2
import numpy as np
import tempfile
import time
from pathlib import Path

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)


# ──────────────────────────────────────────────────────────
# Konstanta
# ──────────────────────────────────────────────────────────
# === MODEL CONFIGURATION ===
MODEL_URL = "https://github.com/get543/Deepfake_Detection_Project/releases/download/v1.0/best_deepfake_model_ff_xception.keras"
# Local cache path (inside your app's working directory)
LOCAL_MODEL_PATH = Path("models") / "best_deepfake_model_ff_xception.keras"
IMG_SIZE        = (224, 224)
FRAMES_TO_SAMPLE = 30
FACE_CASCADE    = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


# ──────────────────────────────────────────────────────────
# Konfigurasi halaman
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepGuard - Deepfake Detection",
    page_icon="🛡️",
    layout="centered",
)

# ──────────────────────────────────────────────────────────
# CSS styling
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #111128 50%, #0d1f2d 100%);
}

header[data-testid="stHeader"] { background: transparent; }

/* Tombol */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.7rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35);
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 28px rgba(99,102,241,0.5);
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #06b6d4) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.3) !important;
    color: #a5b4fc !important;
}

/* Teks umum */
h1, h2, h3 { color: #e2e8f0 !important; }
p, li, label { color: #94a3b8 !important; }
hr { border-color: rgba(255,255,255,0.08) !important; }

/* File uploader */
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(99,102,241,0.4) !important;
    border-radius: 16px !important;
}
</style>
""", unsafe_allow_html=True)



# ──────────────────────────────────────────────────────────
# Fungsi Load Model
# Load model dengan download model dari github releases
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    # Create directory if needed
    LOCAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # If model is not already downloaded, download it
    if not LOCAL_MODEL_PATH.exists():
        with st.spinner("📥 Downloading model from GitHub Releases (this may take a few minutes)..."):
            try:
                response = requests.get(MODEL_URL, stream=True)
                response.raise_for_status()  # Check for download errors
                
                # Save file in chunks to avoid memory issues
                with open(LOCAL_MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("✅ Model downloaded successfully!")
            except Exception as e:
                st.error(f"❌ Failed to download model: {e}")
                return None
    
    # Load the model from local cache
    try:
        model = tf.keras.models.load_model(str(LOCAL_MODEL_PATH))
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None


# ──────────────────────────────────────────────────────────
# Fungsi prediksi — SAMA PERSIS dengan notebook
# (predict_video_wajah: setiap frame ke-5, 1 wajah per frame)
# ──────────────────────────────────────────────────────────

def predict_video_wajah(video_path, model, progress_bar):
    """
    Logika identik dengan notebook:
    - Proses setiap frame ke-5
    - Deteksi wajah (Haar Cascade, 1.3 scale, minNeighbors=5)
    - Potong HANYA wajah, resize ke 224x224
    - Ambil 1 wajah terjelas per frame (break setelah wajah pertama)
    - Tidak ada fallback jika wajah tidak ditemukan
    """
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap         = cv2.VideoCapture(video_path)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps         = cap.get(cv2.CAP_PROP_FPS)
    duration    = total / fps if fps > 0 else 0

    frame_count = 0
    fake_count  = 0
    real_count  = 0
    scores      = []   # skor per wajah yang diproses
    face_images = []   # gambar wajah untuk ditampilkan

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Sama persis dengan notebook: proses setiap frame ke-5
        if frame_count % 5 == 0:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                # Potong HANYA wajah (sama persis notebook, tanpa margin)
                face_img = frame[y:y+h, x:x+w]
                face_img = cv2.resize(face_img, (224, 224))

                face_rgb   = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                face_norm  = face_rgb / 255.0
                face_input = np.expand_dims(face_norm, axis=0)

                # Prediksi
                prediction = model.predict(face_input, verbose=0)[0][0]

                # Keras: fake=0, real=1 → prediction ≈ P(real)
                if prediction > 0.5:
                    real_count += 1
                else:
                    fake_count += 1

                scores.append(float(prediction))
                face_images.append(face_rgb)

                break  # Cukup 1 wajah terjelas per frame (sama dengan notebook)

        frame_count += 1

        # Update progress bar berdasarkan posisi frame
        if total > 0:
            progress_bar.progress(min(frame_count / total, 1.0))

    cap.release()

    total_processed = fake_count + real_count
    if total_processed == 0:
        return None, None, [], [], fps, duration

    fake_prob  = fake_count / total_processed
    avg_score  = float(np.mean(scores))

    label      = "FAKE" if fake_prob > 0.5 else "REAL"
    confidence = fake_prob if label == "FAKE" else (1.0 - fake_prob)

    return label, confidence, scores, face_images, fps, duration


# ──────────────────────────────────────────────────────────
# UI UTAMA
# ──────────────────────────────────────────────────────────

# Header
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1.5rem;">
    <span style="background:rgba(99,102,241,0.15); border:1px solid rgba(99,102,241,0.4);
                 color:#a5b4fc; font-size:0.75rem; font-weight:600; letter-spacing:0.1em;
                 text-transform:uppercase; padding:0.35rem 1rem; border-radius:999px;">
        🛡️ AI-Powered Analysis
    </span>
    <h1 style="font-size:2.8rem; font-weight:800; margin:1rem 0 0.5rem;
               background:linear-gradient(135deg,#e0e7ff,#a5b4fc,#6ee7f7);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        DeepGuard
    </h1>
    <p style="color:#94a3b8; font-size:1rem; max-width:480px; margin:0 auto;">
        Upload video, klik tombol deteksi, dan AI akan menentukan apakah video tersebut
        <strong style="color:#a5b4fc">Real</strong> atau
        <strong style="color:#f87171">AI-Generated (Deepfake)</strong>.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Cek model ──
model = load_model()
if model is None:
    st.error(
        f"⚠️ Model tidak ditemukan di:\n`{LOCAL_MODEL_PATH}`\n\n"
        "Pastikan `best_deepfake_model.keras` ada di folder `notebooks/models/`."
    )
    st.stop()

# ── Upload video ──
st.markdown("### 📁 Upload Video")
uploaded_file = st.file_uploader(
    "Format yang didukung: MP4, AVI, MOV, MKV, WEBM",
    type=["mp4", "avi", "mov", "mkv", "webm"],
)

# ── Tampilkan video preview ──
if uploaded_file:
    st.video(uploaded_file)
    st.markdown("")
    run_detection = st.button("🔍  Detect Deepfake", width="stretch")
else:
    run_detection = False
    st.info("👆 Upload video terlebih dahulu untuk memulai deteksi.")

# ── Jalankan deteksi ──
if uploaded_file and run_detection:
    st.divider()
    st.markdown("### ⚙️ Menganalisis Video…")

    # Simpan file ke disk sementara
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        progress_bar = st.progress(0.0, text="Memproses frame…")
        t_start = time.time()

        label, confidence, frame_scores, face_images, fps, duration = predict_video_wajah(
            tmp_path, model, progress_bar
        )

        elapsed = time.time() - t_start
        progress_bar.empty()

        # ── Tampilkan hasil ──
        if label is None:
            st.error("Gagal membaca frame dari video. Coba video lain.")
        else:
            st.divider()
            st.markdown("### 🎯 Hasil Deteksi")

            # Warna & emoji sesuai label
            if label == "REAL":
                color    = "#34d399"
                bg_color = "rgba(16,185,129,0.12)"
                border   = "rgba(16,185,129,0.35)"
                emoji    = "✅"
            else:
                color    = "#f87171"
                bg_color = "rgba(239,68,68,0.12)"
                border   = "rgba(239,68,68,0.35)"
                emoji    = "🚨"

            # Kartu hasil utama
            st.markdown(f"""
            <div style="background:{bg_color}; border:1px solid {border};
                        border-radius:20px; padding:2rem; text-align:center; margin-bottom:1rem;">
                <div style="font-size:2.2rem; font-weight:800; color:{color}; margin-bottom:0.4rem;">
                    {emoji} {label}
                </div>
                <div style="color:#94a3b8; font-size:0.95rem;">
                    Tingkat kepercayaan: <strong style="color:#e2e8f0">{confidence*100:.1f}%</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Statistik singkat dalam 3 kolom
            col1, col2, col3 = st.columns(3)
            n_real = sum(1 for s in frame_scores if s >= 0.5)
            col1.metric("Frame Dianalisis", len(frame_scores))
            col2.metric("Frame Real",       n_real,               delta=None)
            col3.metric("Frame Fake",       len(frame_scores) - n_real, delta=None)

            st.caption(
                f"⏱ Durasi: {duration:.1f}s  ·  🎞 FPS: {fps:.0f}  ·  "
                f"⚡ Waktu analisis: {elapsed:.1f}s"
            )

            # ── Tabs: Grafik | Sampel Wajah ──
            st.divider()
            tab_chart, tab_faces = st.tabs(["📊 Skor Per-Frame", "🖼️ Sampel Wajah"])

            with tab_chart:
                import pandas as pd
                st.markdown(
                    "<p style='font-size:0.85rem;color:#64748b;'>"
                    "Skor mendekati 1.0 = Real &nbsp;·&nbsp; mendekati 0.0 = Fake</p>",
                    unsafe_allow_html=True,
                )
                chart_df = pd.DataFrame({
                    "Frame":      list(range(1, len(frame_scores) + 1)),
                    "Real Score": frame_scores,
                }).set_index("Frame")
                st.line_chart(chart_df, height=220)

            with tab_faces:
                st.markdown(
                    "<p style='font-size:0.85rem;color:#64748b;'>5 sampel wajah yang dianalisis oleh model.</p>",
                    unsafe_allow_html=True,
                )
                cols = st.columns(5)
                for i, col in enumerate(cols):
                    if i < len(face_images):
                        s       = frame_scores[i]
                        verdict = "Real" if s >= 0.5 else "Fake"
                        c       = "#34d399" if s >= 0.5 else "#f87171"
                        col.image(face_images[i], width="stretch")
                        col.markdown(
                            f"<p style='text-align:center;color:{c};font-size:0.75rem;margin-top:-6px;'>"
                            f"{verdict} ({s:.2f})</p>",
                            unsafe_allow_html=True,
                        )

    finally:
        os.unlink(tmp_path)

# ── Footer ──
st.divider()
st.markdown("""
<p style="text-align:center; color:#334155; font-size:0.78rem;">
    DeepGuard · MobileNetV2 · FaceForensics++ · Penulisan Ilmiah — Universitas Gunadarma
</p>
""", unsafe_allow_html=True)
