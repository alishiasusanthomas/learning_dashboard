import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from utils.audio import extract_features


def load_data(data_folder):
    X = []
    y = []

    for file in os.listdir(data_folder):
        if file.endswith(".wav"):
            file_path = os.path.join(data_folder, file)

            features = extract_features(file_path)
            normalized_name = file.lower()

            # Label based on filename
            if "teacher" in normalized_name:
                label = "teacher"
            elif "student1" in normalized_name:
                label = "student1"
            elif "student2" in normalized_name:
                label = "student2"
            elif "student3" in normalized_name:
                label = "student3"
            else:
                continue

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


def train_model(data_folder):
    X, y = load_data(data_folder)
    if len(X) == 0:
        raise ValueError("No labeled training audio was found in the data folder.")

    n_neighbors = min(3, len(X))

    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights="distance")
    model.fit(X, y)

    return model


def predict_speaker(model, file_path, confidence_threshold=0.5):
    features = extract_features(file_path).reshape(1, -1)
    probabilities = model.predict_proba(features)[0]
    best_index = int(np.argmax(probabilities))
    best_score = float(probabilities[best_index])

    if best_score < confidence_threshold:
        return "Unknown"

    return model.classes_[best_index]
