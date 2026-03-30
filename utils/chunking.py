import librosa


def split_audio(file_path, chunk_duration=3):
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be greater than 0 seconds.")

    audio, sr = librosa.load(file_path, sr=None)

    if audio.size == 0:
        raise ValueError("The audio file is empty.")

    chunk_length = int(sr * chunk_duration)
    if chunk_length <= 0:
        raise ValueError("Unable to create chunks with the current sample rate.")

    chunks = []

    for i in range(0, len(audio), chunk_length):
        chunk = audio[i : i + chunk_length]
        if len(chunk) > 0:
            chunks.append(chunk)

    return chunks, sr
