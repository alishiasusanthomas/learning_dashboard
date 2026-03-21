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

            # Label based on filename
            if "teacher" in file:
                label = "teacher"
            elif "student1" in file:
                label = "student1"
            elif "student2" in file:
                label = "student2"
            elif "student3" in file:
                label = "student3"
            else:
                continue

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)


def train_model(data_folder):
    X, y = load_data(data_folder)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X, y)

    return model


def predict_speaker(model, file_path):
    features = extract_features(file_path).reshape(1, -1)
    prediction = model.predict(features)

    return prediction[0]
