import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

import streamlit as st

from utils.clustering_model import cluster_speakers, get_features
from utils.chunking import split_audio
from utils.resemblyzer_model import (
    enroll_speakers,
    get_resemblyzer_error,
    is_resemblyzer_available,
    match_speaker,
)
from utils.video import extract_audio_from_video

AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv"}
UPLOAD_TYPES = sorted(
    extension.removeprefix(".") for extension in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
)


def format_seconds_label(seconds):
    total_seconds = max(int(round(seconds)), 0)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def format_speaker_name(speaker):
    if not speaker or speaker == "Unknown":
        return "Unknown"
    if speaker == "No speech":
        return "No speech"

    speaker_text = str(speaker)
    if speaker_text.startswith("Speaker_"):
        suffix = speaker_text.split("_", 1)[1]
        if suffix.isdigit():
            return f"Speaker Cluster {int(suffix) + 1}"
        return speaker_text.replace("_", " ")
    return speaker_text.title()


def normalize_cluster_labels(labels):
    label_map = {}
    normalized = []
    next_label = 0
    for label in labels:
        if label not in label_map:
            label_map[label] = next_label
            next_label += 1
        normalized.append(label_map[label])
    return normalized


def compute_chunk_activity(chunks, silence_percentile, silence_scale):
    if not chunks:
        return [], 0.0

    energies = []
    for chunk in chunks:
        squared_sum = sum(float(sample) * float(sample) for sample in chunk)
        energy = (squared_sum / max(len(chunk), 1)) ** 0.5
        energies.append(energy)

    sorted_energies = sorted(energies)
    percentile_index = min(
        int((len(sorted_energies) - 1) * (silence_percentile / 100.0)),
        len(sorted_energies) - 1,
    )
    baseline = sorted_energies[percentile_index]
    threshold = baseline * silence_scale
    activity_mask = [energy >= threshold and energy > 0 for energy in energies]
    return activity_mask, threshold


def build_chunk_rows(chunks, sr, chunk_predictions, similarity_scores):
    rows = []
    current_start = 0.0
    for index, chunk in enumerate(chunks):
        duration = len(chunk) / sr
        current_end = current_start + duration
        score = similarity_scores[index] if index < len(similarity_scores) else None
        rows.append(
            {
                "start_seconds": current_start,
                "end_seconds": current_end,
                "speaker": chunk_predictions[index],
                "score": score,
            }
        )
        current_start = current_end
    return rows


def summarize_speaker_windows(chunk_rows):
    speaker_windows = {}
    for row in chunk_rows:
        speaker = row["speaker"]
        if speaker in {"Unknown", "No speech"}:
            continue
        speaker_windows.setdefault(speaker, []).append(
            f"{format_seconds_label(row['start_seconds'])} - {format_seconds_label(row['end_seconds'])}"
        )
    return speaker_windows


def summarize_speaker_stats(chunk_rows):
    speaker_stats = {}
    for row in chunk_rows:
        speaker = row["speaker"]
        if speaker in {"Unknown", "No speech"}:
            continue
        stats = speaker_stats.setdefault(
            speaker,
            {
                "window_count": 0,
                "duration_seconds": 0.0,
            },
        )
        stats["window_count"] += 1
        stats["duration_seconds"] += row["end_seconds"] - row["start_seconds"]
    return speaker_stats


def summarize_minute_windows(chunk_rows):
    if not chunk_rows:
        return []

    minute_groups = {}
    for row in chunk_rows:
        minute_index = int(row["start_seconds"] // 60)
        minute_groups.setdefault(minute_index, []).append(row)

    summaries = []
    for minute_index in sorted(minute_groups):
        rows = minute_groups[minute_index]
        speaker_counts = Counter(
            row["speaker"] for row in rows if row["speaker"] not in {"Unknown", "No speech"}
        )
        speech_rows = [row for row in rows if row["speaker"] != "No speech"]
        if not speech_rows:
            summary_text = "Nobody spoke"
        elif speaker_counts:
            top_speaker, _ = speaker_counts.most_common(1)[0]
            summary_text = f"{format_speaker_name(top_speaker)} spoke most"
            if len(speaker_counts) > 1:
                summary_text += f" with {len(speaker_counts)} active speakers"
        else:
            summary_text = "Speech was detected, but the speaker was unclear"

        summaries.append(
            {
                "label": f"{minute_index}-{minute_index + 1} min",
                "summary": summary_text,
                "speech_windows": len(speech_rows),
            }
        )
    return summaries


def get_data_signature(data_folder):
    if not os.path.exists(data_folder):
        return ()

    signature = []
    for file_name in sorted(os.listdir(data_folder)):
        file_path = os.path.join(data_folder, file_name)
        if os.path.isfile(file_path):
            file_stat = os.stat(file_path)
            signature.append((file_name, file_stat.st_size, int(file_stat.st_mtime)))
    return tuple(signature)


@st.cache_resource(show_spinner=False)
def load_enrolled_speakers(data_folder, data_signature, reference_duration_seconds):
    del data_signature
    return enroll_speakers(
        data_folder, reference_duration_seconds=reference_duration_seconds
    )


def save_uploaded_file(uploaded_file, destination):
    with open(destination, "wb") as file_obj:
        file_obj.write(uploaded_file.getbuffer())


def prepare_audio_file(uploaded_file, temp_dir):
    suffix = Path(uploaded_file.name).suffix.lower()

    if suffix in AUDIO_EXTENSIONS:
        audio_path = os.path.join(temp_dir, f"uploaded_audio{suffix}")
        save_uploaded_file(uploaded_file, audio_path)
        return audio_path, "audio"

    if suffix in VIDEO_EXTENSIONS:
        video_path = os.path.join(temp_dir, f"uploaded_video{suffix}")
        audio_path = os.path.join(temp_dir, "extracted_audio.wav")
        save_uploaded_file(uploaded_file, video_path)
        extract_audio_from_video(video_path, audio_path)
        return audio_path, "video"

    raise ValueError("Unsupported file format.")


def get_speaker_predictions(
    chunks,
    sr,
    active_mask,
    enrolled_speakers,
    confidence_threshold,
    progress_bar=None,
    progress_text=None,
):
    if not chunks:
        return [], []

    predictions = []
    similarity_scores = []
    total_chunks = len(chunks)
    for index, (chunk, is_active) in enumerate(zip(chunks, active_mask), start=1):
        if not is_active:
            speaker, score = "No speech", 0.0
        else:
            speaker, score = match_speaker(
                chunk,
                sr,
                enrolled_speakers,
                similarity_threshold=confidence_threshold,
            )
        predictions.append(speaker)
        similarity_scores.append(score)
        if progress_bar is not None:
            progress_bar.progress(index / total_chunks)
        if progress_text is not None:
            progress_text.info(f"Matching speakers... ({index}/{total_chunks})")
    return predictions, similarity_scores


def get_cluster_predictions(chunks, sr, active_mask, requested_speakers, progress_text=None):
    if not chunks:
        return []

    active_chunks = [chunk for chunk, is_active in zip(chunks, active_mask) if is_active]
    if not active_chunks:
        return ["No speech"] * len(chunks)

    if len(active_chunks) == 1 or requested_speakers <= 1:
        active_predictions = ["Speaker_0"] * len(active_chunks)
    else:
        feature_vectors = get_features(active_chunks, sr)
        num_speakers = max(1, min(requested_speakers, len(active_chunks)))

        if progress_text is not None:
            progress_text.info(f"Clustering speakers into {num_speakers} groups...")

        labels = cluster_speakers(feature_vectors, num_speakers=num_speakers)
        normalized_labels = normalize_cluster_labels(labels)
        active_predictions = [f"Speaker_{label}" for label in normalized_labels]

    predictions = []
    active_index = 0
    for is_active in active_mask:
        if is_active:
            predictions.append(active_predictions[active_index])
            active_index += 1
        else:
            predictions.append("No speech")
    return predictions


st.set_page_config(page_title="Learner Dashboard", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --app-text: #18212f;
        --app-muted: #4b5563;
        --app-accent: #7c4a1d;
        --app-panel: rgba(255, 255, 255, 0.9);
    }
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(217, 119, 6, 0.12), transparent 30%),
            linear-gradient(180deg, #fff8ef 0%, #f6efe4 100%);
        color: var(--app-text);
    }
    .stApp, .stApp p, .stApp li, .stApp label, .stApp span, .stApp div {
        color: var(--app-text);
    }
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: var(--app-text);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5eadb 0%, #efe2d0 100%);
    }
    [data-testid="stSidebar"] * {
        color: var(--app-text);
    }
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] label,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stCaptionContainer"],
    [data-testid="stAlertContainer"] {
        color: var(--app-text);
    }
    [data-testid="stFileUploaderDropzone"] * {
        color: #ffffff !important;
    }
    [data-testid="stFileUploaderDropzone"] svg {
        fill: #ffffff !important;
    }
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p {
        color: #ffffff !important;
    }
    [data-testid="stBaseButton-primary"] {
        color: #fffaf3;
    }
    [data-testid="stSliderTickBarMin"],
    [data-testid="stSliderTickBarMax"] {
        background-color: rgba(124, 74, 29, 0.18);
    }
    .hero-box, .panel-box {
        background: var(--app-panel);
        border: 1px solid rgba(120, 86, 43, 0.12);
        border-radius: 24px;
        padding: 1.25rem;
        box-shadow: 0 16px 32px rgba(82, 59, 30, 0.08);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 0.35rem;
    }
    .hero-copy {
        color: var(--app-muted);
        line-height: 1.6;
        margin-bottom: 0;
    }
    .metric-box {
        border-radius: 20px;
        padding: 1rem;
        background: linear-gradient(160deg, #fff4e8 0%, #ffe0c7 100%);
        border: 1px solid rgba(214, 120, 59, 0.12);
        margin-bottom: 0.75rem;
    }
    .metric-label {
        font-size: 0.9rem;
        color: var(--app-accent);
        font-weight: 700;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        color: #1f2937;
    }
    .minute-pill {
        border-radius: 18px;
        padding: 0.85rem 1rem;
        background: #f8f1e6;
        border: 1px solid rgba(120, 86, 43, 0.10);
        margin-bottom: 0.6rem;
    }
    .timeline-row {
        display: flex;
        gap: 1rem;
        padding: 0.8rem 0;
        border-bottom: 1px solid rgba(120, 86, 43, 0.10);
    }
    .timeline-row:last-child {
        border-bottom: none;
    }
    .timeline-time {
        min-width: 132px;
        font-weight: 700;
        color: var(--app-accent);
    }
    .timeline-speaker {
        font-weight: 700;
        color: #111827;
    }
    .timeline-note {
        color: var(--app-muted);
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="hero-box">
        <div class="hero-title">Learner Dashboard</div>
        <p class="hero-copy">
            Upload a classroom audio or video recording to estimate who spoke, when they spoke,
            and which learner appears most active during the session.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Settings")
    chunk_duration = st.slider("Audio Chunk Duration (seconds)", 1, 10, 2)
    confidence_threshold = st.slider("Speaker Confidence Threshold", 0.0, 1.0, 0.6)
    reference_duration_seconds = st.slider(
        "Reference Audio Duration (seconds)", 5, 30, 15
    )
    fallback_speaker_count = st.slider("Fallback Speaker Groups", 1, 6, 2)
    silence_percentile = st.slider("Silence Baseline Percentile", 5, 50, 20)
    silence_scale = st.slider("Silence Threshold Scale", 0.8, 2.0, 1.2)

enrolled_speakers = None
if not is_resemblyzer_available():
    st.sidebar.error("Resemblyzer is unavailable in this environment.")
    st.sidebar.caption(str(get_resemblyzer_error()))
elif os.path.exists("data") and os.listdir("data"):
    try:
        data_signature = get_data_signature("data")
        with st.spinner("Enrolling reference speakers..."):
            enrolled_speakers = load_enrolled_speakers(
                "data", data_signature, reference_duration_seconds
            )
        st.sidebar.success(f"Enrolled {len(enrolled_speakers)} speakers")
    except Exception as error:
        st.sidebar.warning(f"Speaker enrollment error: {error}")
else:
    st.sidebar.info("Add labeled .wav files in `data/` to enroll known speakers.")

st.subheader("Upload Audio File")
uploaded_file = st.file_uploader(
    "Upload an audio or video file",
    type=UPLOAD_TYPES,
    help="Supported audio: WAV, MP3, M4A, AAC, FLAC, OGG. Supported video: MP4, AVI, MOV, MKV, FLV.",
)
analyze_clicked = False
if uploaded_file is not None:
    st.caption(
        f"Selected file: {uploaded_file.name} ({uploaded_file.size / 1024:.1f} KB)"
    )
    analyze_clicked = st.button("Analyze Recording", type="primary")
else:
    st.caption("Choose a file first, then click Analyze Recording.")

processing_section = st.container()

if uploaded_file is not None and analyze_clicked:
    processing_section.subheader("Processing...")
    progress_text = processing_section.empty()
    progress_bar = processing_section.progress(0)
    with st.spinner("Processing uploaded file..."):
        temp_dir = tempfile.mkdtemp(prefix="learner_dashboard_")
        try:
            try:
                progress_text.info("Preparing uploaded file...")
                audio_path, input_type = prepare_audio_file(uploaded_file, temp_dir)
            except Exception as error:
                st.error(f"Error preparing file: {error}")
                st.stop()

            try:
                progress_text.info("Splitting audio into chunks...")
                chunks, sr = split_audio(audio_path, chunk_duration)
            except Exception as error:
                st.error(f"Error splitting audio: {error}")
                st.stop()

            if not chunks:
                st.error("No audio chunks could be created from the uploaded file.")
                st.stop()

            try:
                active_mask, silence_threshold = compute_chunk_activity(
                    chunks,
                    silence_percentile,
                    silence_scale,
                )
                if enrolled_speakers is not None:
                    progress_text.info("Matching speakers...")
                    speaker_predictions, similarity_scores = get_speaker_predictions(
                        chunks,
                        sr,
                        active_mask,
                        enrolled_speakers,
                        confidence_threshold,
                        progress_bar=progress_bar,
                        progress_text=progress_text,
                    )
                else:
                    speaker_predictions = get_cluster_predictions(
                        chunks,
                        sr,
                        active_mask,
                        fallback_speaker_count,
                        progress_text=progress_text,
                    )
                    similarity_scores = [None] * len(chunks)
                    progress_bar.progress(1.0)
            except Exception as error:
                st.warning(f"Speaker analysis warning: {error}")
                speaker_predictions = ["Unknown"] * len(chunks)
                similarity_scores = [0.0] * len(chunks)
                silence_threshold = 0.0
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            progress_bar.progress(1.0)

    progress_text.success("Processing complete.")

    speaker_counts = Counter(speaker_predictions)
    known_speaker_counts = Counter(
        {
            speaker: count
            for speaker, count in speaker_counts.items()
            if speaker not in {"Unknown", "No speech"}
        }
    )
    visible_counts = known_speaker_counts or Counter(
        {
            speaker: count
            for speaker, count in speaker_counts.items()
            if speaker != "No speech"
        }
    )
    analysis_mode = (
        "Resemblyzer speaker matching"
        if enrolled_speakers is not None
        else "Fallback speaker clustering (anonymous speaker groups)"
    )
    total_chunks = len(chunks)
    silent_chunks = sum(1 for prediction in speaker_predictions if prediction == "No speech")
    most_active = (
        known_speaker_counts.most_common(1)[0]
        if known_speaker_counts
        else ("Unknown", 0)
    )
    chunk_rows = build_chunk_rows(chunks, sr, speaker_predictions, similarity_scores)
    speaker_windows = summarize_speaker_windows(chunk_rows)
    speaker_stats = summarize_speaker_stats(chunk_rows)
    minute_summaries = summarize_minute_windows(chunk_rows)

    detected_speaker_count = len(visible_counts)

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    with metric_col1:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Most Interactive</div>
                <div class="metric-value">{format_speaker_name(most_active[0])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with metric_col2:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Total Windows Used</div>
                <div class="metric-value">{sum(known_speaker_counts.values())}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with metric_col3:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">Detected Speakers</div>
                <div class="metric-value">{detected_speaker_count}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with metric_col4:
        st.markdown(
            f"""
            <div class="metric-box">
                <div class="metric-label">No Speech Windows</div>
                <div class="metric-value">{silent_chunks}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.caption(
        f"Analysis mode: {analysis_mode} | Total chunks: {total_chunks} | Input type: {input_type} | Silence threshold: {silence_threshold:.4f}"
    )
    if enrolled_speakers is None:
        st.info(
            "Speaker Cluster labels are anonymous groups, not confirmed student identities. "
            "If a recording has only one main voice, set Fallback Speaker Groups to 1."
        )

    counts_col, minute_col = st.columns([1, 1], gap="large")
    with counts_col:
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        st.subheader("Speaker Counts")
        if visible_counts:
            for speaker, count in visible_counts.most_common():
                duration_seconds = speaker_stats.get(speaker, {}).get("duration_seconds", 0.0)
                st.write(
                    f"{format_speaker_name(speaker)} -> {count} windows | total speaking time {format_seconds_label(duration_seconds)}"
                )
        else:
            st.write("No speakers detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with minute_col:
        st.markdown('<div class="panel-box">', unsafe_allow_html=True)
        st.subheader("Minute Summary")
        if minute_summaries:
            for summary in minute_summaries:
                st.markdown(
                    f"""
                    <div class="minute-pill">
                        <strong>{summary['label']}</strong><br>
                        {summary['summary']}<br>
                        <span class="timeline-note">{summary['speech_windows']} speaking windows</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No timeline summary is available.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel-box">', unsafe_allow_html=True)
    with st.expander("Speaking Timeline", expanded=False):
        if visible_counts:
            speaker_breakdown = " | ".join(
                (
                    f"{format_speaker_name(speaker)}: "
                    f"{speaker_stats.get(speaker, {}).get('window_count', 0)} windows, "
                    f"{format_seconds_label(speaker_stats.get(speaker, {}).get('duration_seconds', 0.0))} total"
                )
                for speaker, _ in visible_counts.most_common()
            )
            st.markdown(
                f"""
                <div class="minute-pill">
                    <strong>Detected speakers:</strong> {detected_speaker_count}<br>
                    <span class="timeline-note">{speaker_breakdown}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            for speaker, windows in speaker_windows.items():
                window_summary = ", ".join(windows)
                duration_seconds = speaker_stats.get(speaker, {}).get("duration_seconds", 0.0)
                st.markdown(
                    f"""
                    <div class="minute-pill">
                        <strong>{format_speaker_name(speaker)}</strong><br>
                        <span class="timeline-note">{len(windows)} speaking windows</span><br>
                        <span class="timeline-note">Total speaking time: {format_seconds_label(duration_seconds)}</span><br>
                        <span class="timeline-note">{window_summary}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.info("No speakers detected in the timeline.")
        for row in chunk_rows:
            confidence_text = (
                f" | confidence {row['score']:.2f}"
                if row["score"] is not None and row["speaker"] not in {"No speech", "Unknown"}
                else ""
            )
            st.markdown(
                f"""
                <div class="timeline-row">
                    <div class="timeline-time">{format_seconds_label(row['start_seconds'])} - {format_seconds_label(row['end_seconds'])}</div>
                    <div>
                        <div class="timeline-speaker">{format_speaker_name(row['speaker'])}</div>
                        <div class="timeline-note">{'Nobody spoke in this window' if row['speaker'] == 'No speech' else 'Speech detected'}{confidence_text}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
elif uploaded_file is not None:
    processing_section.subheader("Processing...")
    processing_section.info("File is ready. Click `Analyze Recording` to start.")
else:
    processing_section.subheader("Processing...")
    processing_section.info("Upload a file to begin analysis.")
