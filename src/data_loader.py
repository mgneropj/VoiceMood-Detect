import os
import pandas as pd

# Dataset path
DATASET_PATH = os.path.join("dataset", "ravdess")

# Emotion mapping
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


def load_ravdess_dataset(dataset_path=DATASET_PATH):
    """
    Loads the RAVDESS dataset and returns a DataFrame.
    """

    data = []

    actors = sorted(
        [
            folder
            for folder in os.listdir(dataset_path)
            if folder.startswith("Actor_")
        ]
    )

    for actor in actors:
        actor_path = os.path.join(dataset_path, actor)

        for file in os.listdir(actor_path):

            if not file.endswith(".wav"):
                continue

            parts = file.split("-")

            emotion_code = parts[2]

            data.append(
                {
                    "actor": actor,
                    "filename": file,
                    "filepath": os.path.join(actor_path, file),
                    "emotion_code": emotion_code,
                    "emotion": EMOTION_MAP[emotion_code],
                }
            )

    df = pd.DataFrame(data)

    return df


if __name__ == "__main__":

    df = load_ravdess_dataset()

    print("=" * 50)
    print("VoiceMood Detect - Dataset Loader")
    print("=" * 50)

    print(f"Total Audio Files : {len(df)}")
    print(f"Total Actors      : {df['actor'].nunique()}")
    print(f"Total Emotions    : {df['emotion'].nunique()}")

    print("\nEmotion Distribution\n")
    print(df["emotion"].value_counts())

    print("\nFirst 5 Samples\n")
    print(df.head())

    # Save metadata
    os.makedirs("outputs", exist_ok=True)
    df.to_csv("outputs/ravdess_metadata.csv", index=False)

    print("\nMetadata saved to outputs/ravdess_metadata.csv")