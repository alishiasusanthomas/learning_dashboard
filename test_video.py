from utils.video import extract_audio_from_video
from utils.speaker import train_model, predict_speaker

# Step 1: Extract audio
video_path = "data/classroom.mp4"
audio_path = "data/extracted.wav"

extract_audio_from_video(video_path, audio_path)

# Step 2: Train model
model = train_model("data")

# Step 3: Predict speaker
result = predict_speaker(model, audio_path)

print("Detected Speaker:", result)
