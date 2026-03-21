from utils.speaker import train_model, predict_speaker

# Train model
model = train_model("data")

# Test prediction
test_file = "data/student1.wav"  # change if needed

result = predict_speaker(model, test_file)

print("Predicted Speaker:", result)
