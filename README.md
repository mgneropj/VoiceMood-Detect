# 🎤 VoiceMood Detect

An AI-powered Speech Emotion Recognition (SER) web application that predicts human emotions from voice recordings using Machine Learning and Streamlit.

---

## 📌 Overview

VoiceMood Detect analyzes speech from WAV audio files and predicts the speaker's emotion using an SVM (Support Vector Machine) model trained on the **RAVDESS** dataset.

The application provides interactive audio visualizations including waveform, spectrogram, MFCC heatmap, confidence scores, and downloadable prediction reports.

---

## ✨ Features

- 🎤 Upload WAV audio files
- 😊 Speech emotion prediction
- 📊 Confidence score visualization
- 📈 Audio waveform
- 🌈 Spectrogram
- 🔥 MFCC heatmap
- 📋 Audio information
- 📄 Download prediction report (CSV)
- 💻 Interactive Streamlit web application

---

## 🧠 Machine Learning Pipeline

```
Audio (.wav)
        │
        ▼
Preprocessing
        │
        ▼
Feature Extraction
(MFCC, Chroma, Mel Spectrogram,
Spectral Contrast, ZCR, RMS)
        │
        ▼
Feature Scaling
        │
        ▼
Support Vector Machine (SVM)
        │
        ▼
Predicted Emotion
```

---

## 📂 Project Structure

```
VoiceMood-Detect/
│
├── app/
│   └── app.py
│
├── dataset/
│   └── ravdess/
│
├── docs/
│
├── models/
│   ├── svm_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── outputs/
│   ├── features.csv
│   └── ravdess_metadata.csv
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── feature_extraction.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

**Dataset:** RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

Emotions included:

- Neutral
- Calm
- Happy
- Sad
- Angry
- Fearful
- Disgust
- Surprised

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Librosa
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Joblib

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/VoiceMood-Detect.git
```

Move into the project:

```bash
cd VoiceMood-Detect
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app/app.py
```

---

## 📈 Model Performance

| Model | Accuracy |
|--------|---------:|
| Random Forest | 54.51% |
| Support Vector Machine (SVM) | 65.28% |

---

## 🖥️ Application Features

The web application provides:

- Upload WAV audio
- Audio playback
- Audio metadata
- Waveform visualization
- Spectrogram visualization
- MFCC heatmap
- Emotion prediction
- Confidence score chart
- CSV report download

---

## 📷 Screenshots

Add screenshots after deployment.

Example:

```
docs/home.png

docs/waveform.png

docs/spectrogram.png

docs/prediction.png
```

---

## 🔮 Future Improvements

- CNN-based Speech Emotion Recognition
- LSTM model
- Real-time microphone prediction
- Multi-language emotion recognition
- Model comparison dashboard
- Cloud deployment

---

## 👨‍💻 Author

**Rushikesh Kolhe**

B.Tech – Artificial Intelligence & Data Science

GitHub: https://github.com/mgneropj

LinkedIn: https://www.linkedin.com/in/rushikesh-kolhe-174177292/

---

## ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub.
