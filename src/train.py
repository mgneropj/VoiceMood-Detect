import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# Project Paths
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FEATURES_PATH = os.path.join(BASE_DIR, "outputs", "features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# Load Dataset
# =====================================================
df = pd.read_csv(FEATURES_PATH)

print("=" * 60)
print("Dataset Loaded")
print("=" * 60)
print(df.shape)

# Features & Labels
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# =====================================================
# Encode Labels
# =====================================================
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# =====================================================
# Scale Features
# =====================================================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =====================================================
# Train/Test Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# =====================================================
# Train SVM
# =====================================================
print("\nTraining SVM...\n")

model = SVC(
    kernel="rbf",
    C=10,
    gamma="scale",
    probability=True,   # Enables confidence scores
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# Evaluation
# =====================================================
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("=" * 60)
print(f"SVM Accuracy : {accuracy * 100:.2f}%")
print("=" * 60)

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        target_names=encoder.classes_
    )
)

# =====================================================
# Save Model
# =====================================================
joblib.dump(model, os.path.join(MODEL_DIR, "svm_model.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

print("\nModel Saved Successfully!")
print("Location :", MODEL_DIR)