# 🚀 Learner Dashboard - Setup & Run Guide

## ✅ Project Complete!

Your Learner Dashboard is now fully set up with video upload and analysis capabilities.

---

## 📋 What's Included

### Core Features
✅ **Video Upload** - Upload videos in multiple formats (MP4, AVI, MOV, MKV, FLV)
✅ **Auto Audio Extraction** - Automatically extract audio from videos
✅ **Audio Analysis** - Split audio into chunks and extract features
✅ **Visual Dashboard** - Beautiful Streamlit interface with charts and statistics
✅ **Result Export** - Download analysis reports

### Files Created/Updated
- `app.py` - Main Streamlit dashboard application
- `config.py` - Configuration constants
- `requirements.txt` - Python dependencies
- `utils/__init__.py` - Package initialization
- Updated `README.md` with new instructions

---

## 🔧 Installation

### Step 1: Install Dependencies
Open PowerShell in the project directory and run:

```powershell
pip install -r requirements.txt
```

**Note:** This installs:
- streamlit (dashboard framework)
- librosa (audio processing)
- moviepy (video processing)
- scikit-learn (machine learning)
- numpy, pandas (data processing)
- matplotlib (visualization)

### Step 2: Verify Installation
```powershell
python -c "import streamlit; import librosa; import moviepy; print('✅ All dependencies installed!')"
```

---

## 🎬 Running the Dashboard

### Start the Application
```powershell
streamlit run app.py
```

The dashboard will open in your browser at: `http://localhost:8501`

### Using the Dashboard
1. **Upload a Video**
   - Click the file uploader in the left panel
   - Select a video file (MP4, AVI, MOV, MKV, FLV)
   - The system will process it automatically

2. **View Results**
   - See extracted chunks count
   - View audio statistics
   - Examine MFCC features with visualization
   - See detailed chunk-by-chunk metrics in a table

3. **Download Report**
   - Click "Download Report" to save analysis results as a text file

---

## 🧪 Running Tests

Test individual modules:

```powershell
# Test audio chunking
python test_chunking.py

# Test video processing
python test_video.py

# Test speaker identification
python test_speaker.py

# Test feature extraction
python test_features.py
```

---

## 📊 Dashboard Capabilities

The dashboard provides:

| Feature | Details |
|---------|---------|
| **Video Upload** | Drag & drop or click to upload videos |
| **Audio Extraction** | Automatic extraction from video files |
| **Chunk Analysis** | Splits audio into configurable duration chunks |
| **Feature Extraction** | MFCC (13 coefficients) + Energy metrics |
| **Visualizations** | Bar charts showing feature distributions |
| **Statistics Table** | Detailed metrics for each audio chunk |
| **Report Export** | Download analysis summary |

---

## ⚙️ Configuration

Edit `config.py` to adjust:
- `CHUNK_DURATION` - Length of audio chunks (default: 2 seconds)
- `N_MFCC` - Number of MFCC coefficients (default: 13)
- `KNN_N_NEIGHBORS` - Neighbors for speaker classification (default: 3)

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'moviepy'"
**Solution:** Run `pip install moviepy` separately if installation fails

### Issue: "libsndfile not found"
**Solution:** Install ffmpeg (required for audio processing)
```powershell
# Install ffmpeg
pip install ffmpeg-python
```

### Issue: Streamlit not opening browser
**Solution:** Manually navigate to `http://localhost:8501` in your browser

---

## 📁 Project Structure

```
learner_dashboard/
├── app.py                    # Main dashboard
├── config.py                 # Configuration constants
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
├── SETUP_GUIDE.md           # This file
│
├── utils/
│   ├── __init__.py          # Package initialization
│   ├── audio.py             # Audio feature extraction
│   ├── video.py             # Video to audio conversion
│   ├── chunking.py          # Audio segmentation
│   └── speaker.py           # Speaker identification
│
├── data/                     # Audio files directory
├── test_*.py                # Test files
└── .gitignore               # Git ignore rules
```

---

## 🎓 How It Works

```
User Upload Video
        ↓
Extract Audio (moviepy)
        ↓
Split into Chunks (librosa)
        ↓
Extract Features (MFCC + Energy)
        ↓
Display Results (Streamlit)
        ↓
Download Report
```

---

## ✨ Next Steps

1. ✅ Install dependencies: `pip install -r requirements.txt`
2. ✅ Run dashboard: `streamlit run app.py`
3. ✅ Upload a video file
4. ✅ View analysis results
5. ✅ Download reports

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the main README.md
3. Check the console output for detailed error messages
4. Verify all dependencies: `pip list`

---

**Happy analyzing! 🎉**
