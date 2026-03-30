from utils.chunking import split_audio
from utils.resemblyzer_model import get_embeddings, cluster_speakers

# Load your test audio
audio_file = "data/classroom_test.wav"

# Split audio into chunks
chunks, sr = split_audio(audio_file, chunk_duration=2)

# Get embeddings
embeddings = get_embeddings(chunks)

# Cluster speakers (change number based on your audio)
labels = cluster_speakers(embeddings, num_speakers=2)

# Count speakers
results = {}

for label in labels:
    speaker = f"Speaker_{label}"
    results[speaker] = results.get(speaker, 0) + 1

print("Speaker Count:", results)

# Find most active speaker
most_active = max(results, key=results.get)
print("Most Interactive:", most_active)
