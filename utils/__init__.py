"""
Learner Dashboard Utilities Package

This package contains utilities for:
- Audio feature extraction
- Video to audio conversion
- Audio chunking/segmentation
- Speaker identification
"""

from .audio import extract_features
from .video import extract_audio_from_video
from .chunking import split_audio
from .speaker import load_data, train_model, predict_speaker

__all__ = [
    "extract_features",
    "extract_audio_from_video",
    "split_audio",
    "load_data",
    "train_model",
    "predict_speaker",
]
