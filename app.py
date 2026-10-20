import os
import tempfile
from pathlib import Path

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "metadata" / "cnn_model_v2.keras"

SR = 16000
DURATION = 4.0
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
TARGET_SAMPLES = int(SR * DURATION)

STRESS_NAMES = [
    "Not Stressed",
    "Low Stress",
    "Medium Stress",
    "High Stress",
]
STRESS_COLORS = {
    "Not Stressed": "#2E8B70",
    "Low Stress": "#5D9B73",
    "Medium Stress": "#D69A35",
    "High Stress": "#C85B58",
}

# ============================================================
# PAGE CONFIG + THEME
# ============================================================
st.set_page_config(
    page_title="MindVibe 2.0 | Voice Stress Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        --mv-forest: #214936;
        --mv-green: #2E8B70;
        --mv-mint: #E7F2EB;
        --mv-cream: #F7F7F0;
        --mv-ink: #1D2B24;
        --mv-muted: #65746B;
        --mv-border: #DCE6DD;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        background: linear-gradient(180deg, #F7F8F3 0%, #F3F6F1 100%);
        color: var(--mv-ink);
    }
    [data-testid="stSidebar"] {
        background: #EAF1E9;
        border-right: 1px solid #DCE6DD;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }
    .hero {
        background: linear-gradient(120deg, #214936 0%, #2F7257 100%);
        padding: 2rem 2.1rem;
        border-radius: 24px;
        color: #FFFFFF;
        margin-bottom: 1.2rem;
        box-shadow: 0 12px 32px rgba(33,73,54,.12);
    }
    .hero-kicker {
        font-size: .78rem;
        letter-spacing: .14em;
        text-transform: uppercase;
        color: #D6E9DB;
        font-weight: 700;
        margin-bottom: .55rem;
    }
    .hero h1 {
        color: #FFFFFF;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(2rem, 4vw, 3.2rem);
        line-height: 1.1;
        margin: 0;
        font-weight: 800;
    }
    .hero p {
        color: #E8F2EA;
        font-size: 1.02rem;
        max-width: 720px;
        margin: .85rem 0 0 0;
    }
    .section-title {
        font-family: 'Manrope', sans-serif;
        font-weight: 800;
        color: #214936;
        font-size: 1.35rem;
        margin: .25rem 0 .2rem 0;
    }
    .section-caption {
        color: #65746B;
        font-size: .92rem;
        margin-bottom: .8rem;
    }
    .soft-card {
        background: #FFFFFF;
        border: 1px solid #DCE6DD;
        border-radius: 18px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 5px 18px rgba(30,55,38,.045);
    }
    .prediction-card {
        background: #FFFFFF;
        border: 1px solid #DCE6DD;
        border-radius: 22px;
        padding: 1.4rem;
        box-shadow: 0 8px 25px rgba(30,55,38,.06);
    }
    .eyebrow {
        color: #65746B;
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .1em;
        font-weight: 700;
    }
    .prediction-name {
        color: #214936;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.5rem, 3vw, 2.2rem);
        font-weight: 800;
        line-height: 1.2;
        margin-top: .4rem;
    }
    .small-note {
        color: #65746B;
        font-size: .84rem;
    }
    .pill {
        display: inline-block;
        background: #E7F2EB;
        color: #214936;
        border-radius: 999px;
        padding: .3rem .65rem;
        font-size: .76rem;
        font-weight: 700;
    }
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DCE6DD;
        padding: 1rem 1.1rem;
        border-radius: 16px;
    }
    div[data-testid="stMetricLabel"] {
        color: #65746B;
    }
    .footer-note {
        border-top: 1px solid #DCE6DD;
        padding-top: 1rem;
        margin-top: 2rem;
        color: #65746B;
        font-size: .82rem;
        line-height: 1.6;
    }
    div.stButton > button, div.stDownloadButton > button {
        border-radius: 12px;
        min-height: 2.8rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 🧠 MindVibe")
    st.caption("Voice-based stress analysis")
    page = st.radio(
        "NAVIGATION",
        ["Analyze Audio", "About the Project"],
        label_visibility="visible",
    )
    st.divider()
    st.markdown("**Analysis setup**")
    st.markdown(
        """
        <div class="small-note">
        <b>Input:</b> Speech audio<br>
        <b>Sample rate:</b> 16 kHz<br>
        <b>Clip length:</b> 4 seconds<br>
        <b>Model:</b> CNN v2
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.caption("Research prototype • Not a diagnostic tool")

# ============================================================
# MODEL
# ============================================================
@st.cache_resource(show_spinner="Loading MindVibe CNN v2…")
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)

if not MODEL_PATH.exists():
    st.error(f"CNN v2 model file was not found:\n\n`{MODEL_PATH}`")
    st.info(
        "Place `cnn_model_v2.keras` in `D:\\MindVibe-2.0\\metadata\\`, "
        "or update MODEL_PATH at the top of app.py to the actual file location."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("MindVibe could not load CNN v2.")
    st.code(str(exc))
    st.stop()

# ============================================================
# AUDIO PREPROCESSING — KEEP MATCHED TO TRAINING PIPELINE
# ============================================================
def audio_to_logmel(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=SR, mono=True)

    if y.size == 0:
        raise ValueError("The uploaded recording contains no readable audio.")

    y, _ = librosa.effects.trim(y, top_db=20)

    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)))
    else:
        y = y[:TARGET_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)[..., np.newaxis]


def make_spectrogram_figure(spectrogram: np.ndarray):
    fig, ax = plt.subplots(figsize=(10, 3.8))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    image = librosa.display.specshow(
        spectrogram[:, :, 0],
        sr=SR,
        hop_length=HOP_LENGTH,
        x_axis="time",
        y_axis="mel",
        cmap="magma",
        ax=ax,
    )
    ax.set_title("Log-mel frequency map", loc="left", fontsize=13, pad=12)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Mel frequency")
    fig.colorbar(image, ax=ax, format="%+2.0f dB", pad=0.02)
    fig.tight_layout()
    return fig


def make_probability_chart(probabilities: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({
        "Stress tier": STRESS_NAMES,
        "Probability (%)": np.asarray(probabilities, dtype=float) * 100,
    })


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">MindVibe 2.0 · Speech Intelligence</div>
      <h1>Understand the patterns<br>behind your voice.</h1>
      <p>Explore acoustic patterns in a short speech sample with a four-tier CNN model.
      Upload a clip or record directly, then review the prediction and class probabilities.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if page == "About the Project":
    st.markdown('<div class="section-title">About MindVibe 2.0</div>', unsafe_allow_html=True)
    st.markdown(
        """
        MindVibe 2.0 is a research prototype that analyzes speech using log-mel
        spectrograms and a convolutional neural network (CNN).

        **The four output tiers**
        - Not Stressed
        - Low Stress
        - Medium Stress
        - High Stress

        **Processing pipeline**
        1. Load speech and resample it to 16 kHz.
        2. Trim leading and trailing silence.
        3. Pad or truncate the signal to four seconds.
        4. Create a 128-bin log-mel spectrogram.
        5. Run CNN inference and display the four output scores.

        **Research limitation:** the model was trained using Guided K-Means
        pseudo-labels. Its output scores reflect the model's class predictions
        and are not clinically validated measurements of a person's stress.
        """
    )
    st.stop()

# ============================================================
# AUDIO INPUT AREA
# ============================================================
st.markdown('<div class="section-title">Your audio lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Choose how to provide a speech sample.</div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.05, 1], gap="large")

with left:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    input_method = st.radio(
        "Audio source",
        ["Upload a file", "Record with microphone"],
        horizontal=False,
        key="audio_source",
    )
    audio_file = None

    if input_method == "Upload a file":
        audio_file = st.file_uploader(
            "Choose an audio file",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            help="WAV is recommended. The model uses a 4-second segment.",
        )
    else:
        audio_file = st.audio_input("Record a short speech sample")

#play uploaded or recorded audio
    if audio_file is not None:
        st.markdown("**Listen to your audio**")
        st.audio(audio_file.getvalue())
        
    st.markdown(
        '<p class="small-note">Tip: speak naturally in a quiet environment. '
        'A short, clear sample is easier to inspect.</p>',
        unsafe_allow_html=True,
    )
    
    analyze_clicked = st.button(
        "✨ Analyze speech",
        type="primary",
        use_container_width=True,
        disabled=(audio_file is None),
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="soft-card">', unsafe_allow_html=True)
    st.markdown("**What happens next?**")
    st.markdown(
        """
        <span class="pill">01 · Prepare</span><br><br>
        Audio is converted into a log-mel spectrogram.<br><br>
        <span class="pill">02 · Analyze</span><br><br>
        CNN v2 returns scores for four output tiers.<br><br>
        <span class="pill">03 · Explore</span><br><br>
        Review the predicted tier, all class scores, and the spectrogram.
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PERSISTENT RESULTS / PROCESSING
# ============================================================
if "mv_result" not in st.session_state:
    st.session_state.mv_result = None

if analyze_clicked and audio_file is not None:
    suffix = ".wav" if input_method == "Record with microphone" else (
        Path(audio_file.name).suffix or ".wav"
    )
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(audio_file.getbuffer())
            temp_path = temp_file.name

        with st.spinner("Preparing audio and analyzing acoustic patterns…"):
            spectrogram = audio_to_logmel(temp_path)
            input_data = np.expand_dims(spectrogram, axis=0)
            probabilities = np.asarray(model.predict(input_data, verbose=0)[0], dtype=float)

        if probabilities.size != len(STRESS_NAMES):
            raise ValueError(
                f"Model returned {probabilities.size} outputs; expected {len(STRESS_NAMES)}."
            )
        if not np.all(np.isfinite(probabilities)):
            raise ValueError("The model returned invalid probability values.")

        # Softmax outputs should sum approximately to 1; normalize for display defensively.
        total = probabilities.sum()
        if total <= 0:
            raise ValueError("The model returned scores that cannot be normalized.")
        probabilities = probabilities / total

        predicted_index = int(np.argmax(probabilities))
        st.session_state.mv_result = {
            "name": STRESS_NAMES[predicted_index],
            "confidence": float(probabilities[predicted_index] * 100),
            "probabilities": probabilities,
            "spectrogram": spectrogram,
            "filename": getattr(audio_file, "name", "Recorded speech"),
            "input_method": input_method,
        }

    except Exception as exc:
        st.error("We couldn't analyze this audio. Try a clear WAV recording or another supported file.")
        with st.expander("Technical details"):
            st.code(str(exc))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

# ============================================================
# RESULTS DASHBOARD
# ============================================================
result = st.session_state.mv_result

if result is not None:
    st.divider()
    st.markdown('<div class="section-title">Your analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">These are model-generated predictions, not a diagnosis.</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns([1.4, 1, 1])
    with c1:
        st.markdown(
            f"""
            <div class="prediction-card">
              <div class="eyebrow">Predicted tier</div>
              <div class="prediction-name">{result["name"]}</div>
              <p class="small-note">Based on the highest CNN output score</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.metric("Top-class score", f'{result["confidence"]:.1f}%')
    with c3:
        st.metric("Audio segment", "4 seconds")

    st.caption(f'Analyzed input: {result["filename"]}')

    chart_col, details_col = st.columns([1.2, 1], gap="large")

    with chart_col:
        st.markdown("#### Four-tier score comparison")
        probability_df = make_probability_chart(result["probabilities"])
        st.bar_chart(
            probability_df.set_index("Stress tier"),
            y="Probability (%)",
            horizontal=True,
            color="#2E8B70",
        )
        st.caption(
            "Scores are relative CNN output probabilities. They are not calibrated "
            "clinical probabilities or a measure of stress severity."
        )

    with details_col:
        st.markdown("#### Class scores")
        for label, probability in zip(STRESS_NAMES, result["probabilities"]):
            pct = float(probability * 100)
            st.markdown(f"**{label}** · {pct:.2f}%")
            st.progress(float(np.clip(probability, 0, 1)))

        st.download_button(
            "⬇️ Download analysis (CSV)",
            data=pd.DataFrame({
                "Stress tier": STRESS_NAMES,
                "Model score": result["probabilities"],
                "Model score (%)": result["probabilities"] * 100,
            }).to_csv(index=False).encode("utf-8"),
            file_name="mindvibe_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with st.expander("Explore the log-mel spectrogram"):
        fig = make_spectrogram_figure(result["spectrogram"])
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption(
            "The spectrogram visualizes frequency energy over time. It does not "
            "independently confirm a person's emotional state or stress level."
        )

    if st.button("↻ Clear result and analyze another sample"):
        st.session_state.mv_result = None
        st.rerun()

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer-note">
    <b>Research prototype notice:</b> MindVibe 2.0 predicts one of four classes
    using a CNN trained with Guided K-Means pseudo-labels. The system has not been
    established as a clinical stress assessment tool. Do not use its output for
    medical, employment, or other high-stakes decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
