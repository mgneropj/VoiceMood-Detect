import os
import librosa
import numpy as np
import pandas as pd

# -------------------------------
# Project Paths
# -------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "ravdess")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "features.csv")

# -------------------------------
# Emotion Mapping
# -------------------------------
EMOTION_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fearful",
    "07": "Disgust",
    "08": "Surprised"
}


def extract_features(file_path):
    """
    Extract audio features from one WAV file.
    """
    signal, sr = librosa.load(file_path, sr=None)

    # MFCC
    mfcc = np.mean(
        librosa.feature.mfcc(
            y=signal,
            sr=sr,
            n_mfcc=40
        ).T,
        axis=0
    )

    # Chroma
    chroma = np.mean(
        librosa.feature.chroma_stft(
            y=signal,
            sr=sr
        ).T,
        axis=0
    )

    # Mel Spectrogram
    mel = np.mean(
        librosa.feature.melspectrogram(
            y=signal,
            sr=sr
        ).T,
        axis=0
    )

    # Spectral Contrast
    contrast = np.mean(
        librosa.feature.spectral_contrast(
            y=signal,
            sr=sr
        ).T,
        axis=0
    )

    # Zero Crossing Rate
    zcr = np.mean(
        librosa.feature.zero_crossing_rate(signal)
    )

    # RMS Energy
    rms = np.mean(
        librosa.feature.rms(y=signal)
    )

    features = np.hstack([
        mfcc,
        chroma,
        mel,
        contrast,
        zcr,
        rms
    ])

    return features


def build_dataset():

    rows = []

    actors = sorted([
        actor
        for actor in os.listdir(DATASET_PATH)
        if actor.startswith("Actor_")
    ])

    total = 0

    for actor in actors:

        actor_path = os.path.join(DATASET_PATH, actor)

        for file in os.listdir(actor_path):

            if not file.endswith(".wav"):
                continue

            emotion = EMOTION_MAP[file.split("-")[2]]

            path = os.path.join(actor_path, file)

            features = extract_features(path)

            row = features.tolist()
            row.append(emotion)

            rows.append(row)

            total += 1

            if total % 100 == 0:
                print(f"Processed {total} files...")

    df = pd.DataFrame(rows)

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nFeature Extraction Completed!")
    print(f"Processed Files : {total}")
    print(f"Saved File      : {OUTPUT_PATH}")


if __name__ == "__main__":
    build_dataset()