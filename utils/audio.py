import librosa
import numpy as np


def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path)

    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    energy = np.mean(librosa.feature.rms(y=audio))

    return np.hstack([mfcc_mean, energy])
