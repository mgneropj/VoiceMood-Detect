import os
import tempfile
import joblib
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st
from datetime import datetime

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="VoiceMood Detect",
    page_icon="🎤",
    layout="wide"
)

st.title("🎤 VoiceMood Detect")
st.markdown("### AI Powered Speech Emotion Recognition")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("📌 Project Information")

st.sidebar.success("Model : Support Vector Machine (SVM)")
st.sidebar.info("Dataset : RAVDESS")
st.sidebar.write("Accuracy : **65.28%**")
st.sidebar.write("Features :")
st.sidebar.write("- MFCC")
st.sidebar.write("- Chroma")
st.sidebar.write("- Mel Spectrogram")
st.sidebar.write("- Spectral Contrast")
st.sidebar.write("- Zero Crossing Rate")
st.sidebar.write("- RMS Energy")

# =====================================================
# PATHS
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "svm_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
encoder = joblib.load(ENCODER_PATH)

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def extract_features(file_path):

    signal, sr = librosa.load(file_path, sr=None)

    mfcc = np.mean(
        librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=40
        ).T,
        axis=0
    )

    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=signal,
            sr=sr
        ).T,
        axis=0
    )

    mel = np.mean(
        librosa.feature.melspectrogram(
            y=signal,
            sr=sr
        ).T,
        axis=0
    )

    contrast = np.mean(
        librosa.feature.spectral_contrast(
            y=signal,
            sr=sr
        ).T,
        axis=0
    )

    zcr = np.mean(
        librosa.feature.zero_crossing_rate(signal)
    )

    rms = np.mean(
        librosa.feature.rms(y=signal)
    )

    return np.hstack([
        mfcc,
        chroma,
        mel,
        contrast,
        zcr,
        rms
    ])

# =====================================================
# FILE UPLOAD
# =====================================================
uploaded_file = st.file_uploader(
    "📂 Upload WAV File",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:

        tmp.write(uploaded_file.read())

        temp_path = tmp.name

    signal, sr = librosa.load(temp_path, sr=None)

    duration = librosa.get_duration(
        y=signal,
        sr=sr
    )

    # =====================================================
    # AUDIO INFORMATION
    # =====================================================
    st.subheader("🎵 Audio Information")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Sample Rate", f"{sr} Hz")

    with c2:
        st.metric("Duration", f"{duration:.2f} sec")

    with c3:
        st.metric("Prediction Time",
                  datetime.now().strftime("%H:%M:%S"))

    # =====================================================
    # TWO COLUMN LAYOUT
    # =====================================================

    left, right = st.columns(2)

    # =====================================================
    # WAVEFORM
    # =====================================================

    with left:

        st.subheader("📈 Waveform")

        fig1, ax1 = plt.subplots(figsize=(7,3))

        librosa.display.waveshow(
            signal,
            sr=sr,
            ax=ax1
        )

        st.pyplot(fig1)

        plt.close(fig1)

    # =====================================================
    # SPECTROGRAM
    # =====================================================

    with right:

        st.subheader("🌈 Spectrogram")

        D = librosa.amplitude_to_db(
            np.abs(librosa.stft(signal)),
            ref=np.max
        )

        fig2, ax2 = plt.subplots(figsize=(7,3))

        img = librosa.display.specshow(
            D,
            sr=sr,
            x_axis="time",
            y_axis="log",
            ax=ax2
        )

        fig2.colorbar(
            img,
            ax=ax2,
            format="%+2.0f dB"
        )

        st.pyplot(fig2)

        plt.close(fig2)
            # =====================================================
    # MFCC HEATMAP
    # =====================================================

    st.subheader("🔥 MFCC Heatmap")

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sr,
        n_mfcc=40
    )

    fig3, ax3 = plt.subplots(figsize=(12,4))

    sns.heatmap(
        mfcc,
        cmap="viridis",
        ax=ax3,
        cbar=True
    )

    ax3.set_xlabel("Time Frames")
    ax3.set_ylabel("MFCC Coefficients")
    ax3.set_title("MFCC Heatmap")

    st.pyplot(fig3)

    plt.close(fig3)

    # =====================================================
    # FEATURE EXTRACTION
    # =====================================================

    features = extract_features(temp_path)

    features = scaler.transform([features])

    # =====================================================
    # PREDICTION
    # =====================================================

    prediction = model.predict(features)

    emotion = encoder.inverse_transform(prediction)[0]

    st.subheader("😊 Predicted Emotion")

    st.success(f"### {emotion}")

    # =====================================================
    # CONFIDENCE SCORES
    # =====================================================

    st.subheader("📊 Confidence Scores")

    try:

        probabilities = model.predict_proba(features)[0]

        confidence = pd.DataFrame({
            "Emotion": encoder.classes_,
            "Confidence (%)": probabilities * 100
        })

        confidence = confidence.sort_values(
            by="Confidence (%)",
            ascending=False
        )

        st.bar_chart(
            confidence.set_index("Emotion")
        )

        st.dataframe(
            confidence.style.format({
                "Confidence (%)": "{:.2f}"
            }),
            use_container_width=True
        )

    except Exception:

        st.info(
            "Confidence scores are unavailable for this model."
        )

    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    st.subheader("📄 Prediction Report")

    report = pd.DataFrame({

        "File":[uploaded_file.name],

        "Predicted Emotion":[emotion],

        "Sample Rate":[sr],

        "Duration":[round(duration,2)],

        "Prediction Time":[
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ]

    })

    csv = report.to_csv(index=False)

    st.download_button(

        "⬇ Download Report",

        csv,

        file_name="prediction_report.csv",

        mime="text/csv"

    )

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    st.markdown("---")

    st.subheader("📈 Model Information")

    model_info = pd.DataFrame({

        "Model":[
            "Random Forest",
            "Support Vector Machine"
        ],

        "Accuracy":[
            "54.51%",
            "65.28%"
        ]

    })

    st.table(model_info)

    st.success("✅ Prediction Completed Successfully!")