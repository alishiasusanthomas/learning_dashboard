import os
from collections import defaultdict

import librosa
import numpy as np
from sklearn.cluster import KMeans

_IMPORT_ERROR = None
_ENCODER = None

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
except Exception as error:
    VoiceEncoder = None
    preprocess_wav = None
    _IMPORT_ERROR = error


def is_resemblyzer_available():
    return _IMPORT_ERROR is None


def get_resemblyzer_error():
    return _IMPORT_ERROR


def get_encoder():
    global _ENCODER

    if not is_resemblyzer_available():
        raise RuntimeError(f"Resemblyzer is unavailable: {_IMPORT_ERROR}")

    if _ENCODER is None:
        _ENCODER = VoiceEncoder()

    return _ENCODER


def _normalize_embedding(embedding):
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def extract_speaker_label(file_name):
    stem = os.path.splitext(file_name)[0]
    cleaned = stem.replace("_", " ").replace("-", " ").strip()
    return cleaned or stem


def is_reference_speaker_file(file_name):
    return file_name.lower().endswith(".wav")


def embed_audio_file(file_path, max_duration_seconds=20):
    encoder = get_encoder()
    wav, sample_rate = librosa.load(
        file_path, sr=None, mono=True, duration=max_duration_seconds
    )
    wav = preprocess_wav(wav, source_sr=sample_rate)
    embedding = encoder.embed_utterance(wav)
    return _normalize_embedding(embedding)


def embed_audio_chunk(chunk, sample_rate):
    encoder = get_encoder()
    wav = preprocess_wav(chunk, source_sr=sample_rate)
    embedding = encoder.embed_utterance(wav)
    return _normalize_embedding(embedding)


def enroll_speakers(data_folder, reference_duration_seconds=20):
    speaker_embeddings = defaultdict(list)

    for file_name in os.listdir(data_folder):
        if not is_reference_speaker_file(file_name):
            continue

        file_path = os.path.join(data_folder, file_name)
        speaker_label = extract_speaker_label(file_name)
        speaker_embeddings[speaker_label].append(
            embed_audio_file(
                file_path, max_duration_seconds=reference_duration_seconds
            )
        )

    if not speaker_embeddings:
        raise ValueError("No .wav files were found for speaker enrollment.")

    enrolled = {}
    for speaker_label, embeddings in speaker_embeddings.items():
        enrolled[speaker_label] = _normalize_embedding(np.mean(embeddings, axis=0))

    return enrolled


def match_speaker(chunk, sample_rate, enrolled_speakers, similarity_threshold=0.6):
    if not enrolled_speakers:
        return "Unknown", 0.0

    chunk_embedding = embed_audio_chunk(chunk, sample_rate)
    best_speaker = "Unknown"
    best_score = -1.0

    for speaker_label, speaker_embedding in enrolled_speakers.items():
        score = float(np.dot(chunk_embedding, speaker_embedding))
        if score > best_score:
            best_score = score
            best_speaker = speaker_label

    if best_score < similarity_threshold:
        return "Unknown", best_score

    return best_speaker, best_score


def get_embeddings(wav_slices):
    encoder = get_encoder()
    embeddings = []
    for wav in wav_slices:
        emb = encoder.embed_utterance(wav)
        embeddings.append(emb)
    return np.array(embeddings)


def cluster_speakers(embeddings, num_speakers=2):
    kmeans = KMeans(n_clusters=num_speakers, random_state=0)
    labels = kmeans.fit_predict(embeddings)
    return labels
