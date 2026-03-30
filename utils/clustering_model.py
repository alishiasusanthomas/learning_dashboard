from sklearn.cluster import KMeans
import numpy as np
import librosa


def extract_features_from_chunk(chunk, sr):
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)


def get_features(chunks, sr):
    features = []

    for chunk in chunks:
        feat = extract_features_from_chunk(chunk, sr)
        features.append(feat)

    return np.array(features)


def cluster_speakers(features, num_speakers=2):
    kmeans = KMeans(n_clusters=num_speakers, random_state=0)
    labels = kmeans.fit_predict(features)
    return labels
