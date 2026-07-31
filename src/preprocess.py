import os

import librosa
import librosa.display
import matplotlib.pyplot as plt

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUDIO_FILE = os.path.join(
    BASE_DIR,
    "dataset",
    "ravdess",
    "Actor_01",
    "03-01-03-01-01-01-01.wav",
)


def load_audio(file_path):
    """
    Load an audio file.
    """
    signal, sample_rate = librosa.load(file_path, sr=None)

    print("=" * 50)
    print("Audio Information")
    print("=" * 50)
    print(f"File Name   : {os.path.basename(file_path)}")
    print(f"Sample Rate : {sample_rate} Hz")
    print(f"Duration    : {librosa.get_duration(y=signal, sr=sample_rate):.2f} seconds")
    print(f"Samples     : {len(signal)}")

    return signal, sample_rate


def plot_waveform(signal, sample_rate):
    plt.figure(figsize=(12, 4))
    librosa.display.waveshow(signal, sr=sample_rate)
    plt.title("Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()


def plot_spectrogram(signal, sample_rate):
    plt.figure(figsize=(12, 5))

    spectrogram = librosa.amplitude_to_db(
        abs(librosa.stft(signal)),
        ref=max
    )

    librosa.display.specshow(
        spectrogram,
        sr=sample_rate,
        x_axis="time",
        y_axis="hz",
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title("Spectrogram")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    signal, sr = load_audio(AUDIO_FILE)

    plot_waveform(signal, sr)

    plot_spectrogram(signal, sr)