import librosa


def split_audio(file_path, chunk_duration=3):
    audio, sr = librosa.load(file_path)

    chunk_length = int(sr * chunk_duration)
    chunks = []

    for i in range(0, len(audio), chunk_length):
        chunk = audio[i : i + chunk_length]
        chunks.append(chunk)

    return chunks, sr
