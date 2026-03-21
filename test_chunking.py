from utils.chunking import split_audio
from utils.speaker import train_model, predict_speaker
import soundfile as sf
import os

# Train model
model = train_model("data")

# Input audio (use any long audio)
audio_file = "data/classroom_test.wav"  # or your classroom audio

chunks, sr = split_audio(audio_file)

results = {}

# Create temp folder
if not os.path.exists("temp"):
    os.makedirs("temp")

for i, chunk in enumerate(chunks):
    temp_file = f"temp/chunk_{i}.wav"

    # Save chunk
    sf.write(temp_file, chunk, sr)

    # Predict speaker
    speaker = predict_speaker(model, temp_file)

    print(f"Chunk {i}: {speaker}")

    # Count frequency
    if speaker in results:
        results[speaker] += 1
    else:
        results[speaker] = 1

print("\nSpeaker Count:")
print(results)

# Find most interactive student
most_active = max(results, key=results.get)
print("\nMost Interactive:", most_active)
