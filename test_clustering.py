from utils.chunking import split_audio
from utils.clustering_model import get_features, cluster_speakers

audio_file = "data/classroom_test.wav"

chunks, sr = split_audio(audio_file, chunk_duration=3)

features = get_features(chunks, sr)

labels = cluster_speakers(features, num_speakers=2)

results = {}

for label in labels:
    speaker = f"Speaker_{label}"
    results[speaker] = results.get(speaker, 0) + 1

print("Speaker Count:", results)

most_active = max(results, key=results.get)
print("Most Interactive:", most_active)
