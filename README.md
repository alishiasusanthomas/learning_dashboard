# 🎓 Learner Identification & Engagement Dashboard

## 📌 Overview

This project focuses on analyzing online classroom audio to identify speakers (teacher and students) and determine student engagement levels. The system uses audio processing and machine learning techniques to classify speakers and evaluate participation based on voice activity.

---

## 🎯 Objectives

* Identify speakers (Teacher / Students) using voice data
* Analyze student engagement based on speaking activity
* Build a simple dashboard to visualize engagement
* Demonstrate audio-based learning analytics

---

## 🧠 Problem Statement

In online learning environments, it is difficult for instructors to track which students are actively participating. This project aims to solve this problem by analyzing voice interactions to identify learners and measure their engagement.

---

## ⚙️ System Architecture

```
Audio Input → Feature Extraction → Machine Learning Models → Dashboard Output
```

---

## 🔍 Key Features

### 🎤 1. Audio Processing

* Input: Recorded classroom audio
* Format: `.wav`
* Preprocessing using Librosa

---

### 📊 2. Feature Extraction

* MFCC (Mel-Frequency Cepstral Coefficients)
* Energy (RMS)

These features help in capturing:

* Voice characteristics (speaker identity)
* Activity level (engagement)

---

### 🤖 3. Speaker Identification

* Model: K-Nearest Neighbors (KNN)
* Classifies:

  * Teacher
  * Student 1
  * Student 2
  * Student 3

---

### 📈 4. Engagement Detection

* Based on:

  * Speaking frequency
  * Audio energy
* Output:

  * Engaged ✅
  * Not Engaged ❌

---

### 📊 5. Dashboard (Streamlit)

* **Video Upload:** Upload video files in multiple formats (MP4, AVI, MOV, MKV, FLV)
* **Auto Audio Extraction:** Automatically extracts audio from uploaded videos
* **Real-time Processing:** Analyzes audio chunks and extracts features instantly
* **Visual Reports:** Clear charts and detailed statistics of audio analysis
* **Downloadable Results:** Export analysis reports for further review
* Displays:

  * Speaker identification
  * Engagement status
  * Participation summary
  * Audio feature visualizations
  * Chunk-by-chunk analysis

---

## 🛠️ Technologies Used

* Python
* Librosa (Audio Processing)
* NumPy, Pandas (Data Processing)
* Scikit-learn (Machine Learning)
* MoviePy (Video Processing)
* Streamlit (Interactive Dashboard)
* Matplotlib (Visualization)

---

## 📁 Project Structure

```
learner_dashboard/
│
├── app.py                 # Streamlit dashboard
├── test.py                # Library testing
├── test_features.py       # Feature extraction testing
│
├── utils/
│   └── audio.py           # Feature extraction logic
│
├── data/                  # Audio files (ignored in Git)
├── requirements.txt
└── .gitignore
```

---

## 🚀 How It Works

1. Audio is collected from teacher and students
2. Features (MFCC + Energy) are extracted using Librosa
3. Machine learning model is trained on extracted features
4. New audio is classified into speaker categories
5. Engagement is evaluated based on activity
6. Results are displayed on a dashboard

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run the Interactive Dashboard

```bash
streamlit run app.py
```

Then:
1. Open your browser and navigate to `http://localhost:8501`
2. Upload a video file (MP4, AVI, MOV, MKV, FLV)
3. The dashboard will automatically:
   - Extract audio from the video
   - Split audio into chunks
   - Extract MFCC and energy features
   - Display results with visualizations
   - Provide downloadable analysis reports

### Run Feature Extraction Test

```bash
python test_features.py
```

### Run Individual Module Tests

```bash
python test_chunking.py  # Test audio chunking
python test_video.py     # Test video to audio conversion
python test_speaker.py   # Test speaker identification
```

---

## 📊 Dataset

* Custom dataset created by recording:

  * Teacher voice
  * Multiple student voices
* Audio variations include tone, pitch, and speaking style

---

## ⚠️ Limitations

* Works best in controlled environments
* Limited dataset size
* Not suitable for large-scale real-world deployment

---

## 🔮 Future Improvements

* Use deep learning models for better accuracy
* Real-time audio processing
* Integration with online meeting platforms
* Advanced engagement metrics

---

## 💬 Conclusion

This project demonstrates how audio analysis and machine learning can be used to enhance online learning by identifying learners and measuring their engagement.



## 🚀 Improved Version (Speaker Clustering with Resemblyzer)

In addition to the initial KNN-based speaker identification, the system has been enhanced using a pretrained model (Resemblyzer) to handle unknown speakers.

### 🔍 Improvements:
- Supports **unknown speakers**
- Uses **voice embeddings instead of manual labels**
- Applies **clustering (KMeans)** to group similar voices
- More flexible for real-world scenarios

### ⚙️ Updated Workflow:
Video/Audio → Feature Extraction → Embeddings → Clustering → Speaker Groups → Most Interactive Speaker

### 📊 Output Example:
Speaker_0 → 15  
Speaker_1 → 8  

Most Interactive Speaker → Speaker_0

### 💡 Note:
This approach does not require predefined speaker labels and can generalize to unseen voices.
