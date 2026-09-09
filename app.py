import tempfile
import cv2
import streamlit as st
from pipeline import TrackingPipeline

# 1. UI Setup & Page Configuration
st.set_page_config(page_title="PROGLINT-CV Person Counter", layout="wide")
st.title("🏃 Person Tracker & IN / OUT Counter")

# 2. Sidebar Controls
st.sidebar.header("Configuration")
line_pos = st.sidebar.slider("Counting Line Position Ratio", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
conf_thresh = st.sidebar.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

# 3. Video File Upload Workflow
video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])

if video_file:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_file.getbuffer())
        video_path = f.name

    # Initialize tracking pipeline with YOLOv8, ByteTrack, and PersonCounter
    pipeline = TrackingPipeline("models/best.pt", line_position=line_pos)
    cap = cv2.VideoCapture(video_path)

    video_display = st.empty()
    status_display = st.empty()

    # Frame processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame through detection, tracking, line-crossing counter, and rendering
        annotated_frame, stats = pipeline.process(frame, conf_thresh=conf_thresh)
        video_display.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Display live IN, OUT, INSIDE, Total Unique, and FPS metrics
        status_display.markdown(
            f"📥 **IN: {stats['in']}** | 📤 **OUT: {stats['out']}** | 🚪 **INSIDE: {stats['inside']}** | "
            f"👥 **Total Unique: {stats['total']}** | ⚡ **FPS: {stats['fps']:.1f}**"
        )

    cap.release()
    st.success(
        f"Processing Complete! Final Counts ➔ IN: **{stats['in']}** | OUT: **{stats['out']}** | "
        f"CURRENTLY INSIDE: **{stats['inside']}** | Total Unique People: **{pipeline.total_unique_count}**"
    )
