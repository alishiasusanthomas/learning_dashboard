import streamlit as st
import os
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from utils.video import extract_audio_from_video
from utils.chunking import split_audio
from utils.audio import extract_features
from utils.speaker import load_data, train_model, predict_speaker

# Set up the page
st.set_page_config(page_title="Learner Dashboard", layout="wide")
st.title("🎓 Learner Dashboard - Speaker Recognition")
st.markdown("---")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Settings")
    chunk_duration = st.slider("Audio Chunk Duration (seconds)", 1, 10, 2)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

# Speaker identification model (if training data exists)
model = None
if os.path.exists("data") and os.listdir("data"):
    try:
        with st.spinner("🔄 Training speaker identification model..."):
            model = train_model("data")
        st.sidebar.success("✅ Speaker model trained")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Speaker model training error: {e}")
else:
    st.sidebar.info("ℹ️ Add labeled .wav files in `data/` for speaker identification")

# Create main layout
col1, col2 = st.columns([2, 3])

with col1:
    st.header("📹 Video Upload")
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=["mp4", "avi", "mov", "mkv", "flv"],
        help="Supported formats: MP4, AVI, MOV, MKV, FLV",
    )

# Process the uploaded video
if uploaded_file is not None:
    with st.spinner("🔄 Processing video..."):
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded video
            video_path = os.path.join(temp_dir, "uploaded_video.mp4")
            with open(video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract audio
            audio_path = os.path.join(temp_dir, "extracted_audio.wav")
            st.info("🎵 Extracting audio from video...")
            try:
                extract_audio_from_video(video_path, audio_path)
            except Exception as e:
                st.error(f"Error extracting audio: {str(e)}")
                st.stop()

            # Split audio into chunks
            st.info("✂️ Splitting audio into chunks...")
            try:
                chunks, sr = split_audio(audio_path, chunk_duration)
            except Exception as e:
                st.error(f"Error splitting audio: {str(e)}")
                st.stop()

            # Extract features from audio chunks
            st.info("🔍 Analyzing audio features...")
            features_list = []
            try:
                features = extract_features(audio_path)
                features_list.append(features)
            except Exception as e:
                st.error(f"Error extracting features: {str(e)}")
                st.stop()

            # Speaker prediction (optional)
            speaker_predictions = []
            if model is not None and len(chunks) > 0:
                st.info("🎤 Predicting speaker for each chunk...")
                try:
                    import soundfile as sf
                    from collections import Counter

                    for i, chunk in enumerate(chunks):
                        chunk_path = os.path.join(temp_dir, f"chunk_{i}.wav")
                        sf.write(chunk_path, chunk, sr)
                        speaker = predict_speaker(model, chunk_path)
                        speaker_predictions.append(speaker)
                except Exception as e:
                    st.warning(f"Speaker prediction warning: {e}")
                    speaker_predictions = ["Unknown"] * len(chunks)
            else:
                speaker_predictions = ["Unknown"] * len(chunks)

    # Display Results
    st.markdown("---")
    st.header("📊 Analysis Results")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:
        st.metric("Total Chunks", len(chunks))

    with result_col2:
        st.metric("Sample Rate", f"{sr} Hz")

    with result_col3:
        st.metric("Total Duration", f"{len(chunks) * chunk_duration:.1f}s")

    # Display feature analysis
    st.subheader("🎼 Audio Features")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**MFCC Features (13 coefficients + Energy)**")
        features_array = features_list[0]
        st.info(f"Feature Vector Length: {len(features_array)}")

        # Display feature visualization
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(features_array)), features_array)
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Feature Value")
        ax.set_title("Audio Feature Distribution")
        st.pyplot(fig)

    with col2:
        st.write("**Processing Summary**")
        st.success("✅ Video uploaded successfully")
        st.success("✅ Audio extracted")
        st.success(f"✅ Split into {len(chunks)} chunks")
        st.success("✅ Features extracted")

    # Speaker engagement section
    if speaker_predictions and speaker_predictions.count("Unknown") != len(speaker_predictions):
        st.subheader("👥 Speaker Engagement")
        speaker_counts = Counter(speaker_predictions)
        total_chunks = len(speaker_predictions)
        
        engagement_data = []
        for speaker, count in speaker_counts.most_common():
            engagement_data.append({
                "Speaker": speaker,
                "Chunks": count,
                "Duration (seconds)": count * chunk_duration,
                "Engagement %": f"{(count / total_chunks) * 100:.1f}%",
            })

        st.dataframe(pd.DataFrame(engagement_data), use_container_width=True)
        
        most_active = speaker_counts.most_common(1)[0]
        st.success(f"🏆 Most active speaker: {most_active[0]} ({most_active[1]} chunks, {(most_active[1]/total_chunks)*100:.1f}%)")

    # Detailed chunk information
    st.subheader("📈 Chunk Details")
    chunk_stats = []
    for i, chunk in enumerate(chunks):
        chunk_stats.append(
            {
                "Chunk #": i + 1,
                "Duration (samples)": len(chunk),
                "Duration (seconds)": len(chunk) / sr,
                "Max Amplitude": np.max(np.abs(chunk)),
                "RMS Energy": np.sqrt(np.mean(chunk**2)),
                "Speaker": speaker_predictions[i] if speaker_predictions else "N/A",
            }
        )

    df_chunks = pd.DataFrame(chunk_stats)
    st.dataframe(df_chunks, use_container_width=True)

    # Download results
    st.subheader("📥 Download Results")
    col1, col2 = st.columns(2)

    with col1:
        # Create summary report
        summary_text = f"""
# Audio Analysis Report

## Summary
- Total Chunks: {len(chunks)}
- Sample Rate: {sr} Hz
- Total Duration: {len(chunks) * chunk_duration:.1f} seconds
- Chunk Duration: {chunk_duration} seconds

## Features
- MFCC Coefficients: 13
- Total Features: {len(features_array)}
- Feature Statistics:
  - Mean: {np.mean(features_array):.4f}
  - Std: {np.std(features_array):.4f}
  - Min: {np.min(features_array):.4f}
  - Max: {np.max(features_array):.4f}
"""
        st.download_button(
            label="📄 Download Report",
            data=summary_text,
            file_name="analysis_report.txt",
            mime="text/plain",
        )

# Add information section
st.markdown("---")
st.header("ℹ️ About This Dashboard")
st.info(
    """
**Features:**
- 📹 Upload video files in multiple formats
- 🎵 Automatic audio extraction from videos
- ✂️ Smart audio chunking for analysis
- 🔍 Advanced audio feature extraction (MFCC)
- 📊 Detailed visualization and reports
- 💾 Export analysis results

**Supported Video Formats:** MP4, AVI, MOV, MKV, FLV
"""
)
