"""
Configuration and constants for the Learner Dashboard
"""

# Audio Processing
SAMPLE_RATE = 22050
CHUNK_DURATION = 2  # seconds
MIN_AUDIO_DURATION = 1  # seconds

# Feature Extraction
N_MFCC = 13
MFCC_HOP_LENGTH = 512

# Model Configuration
KNN_N_NEIGHBORS = 3

# Dashboard Settings
STREAMLIT_THEME = "light"
PAGE_LAYOUT = "wide"

# Supported video formats
SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv", "flv", "webm"]

# Speaker Categories
SPEAKER_CATEGORIES = ["teacher", "student1", "student2", "student3"]
