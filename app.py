import os
import tempfile
import cv2
import streamlit as st
from pipeline import TrackingPipeline

# 1. UI Setup & Page Configuration
st.set_page_config(page_title="PROGLINT-CV Person Counter", layout="centered")
st.title("Person Tracker")

# Load external styling from dedicated style.css file
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.html(f"<style>{f.read()}</style>")

# 2. Video File Upload Workflow
video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])

if video_file:
    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_file.getbuffer())
        video_path = f.name

    # Initialize tracking pipeline
    pipeline = TrackingPipeline("models/mot17_best.pt", line_position=0.5)
    cap = cv2.VideoCapture(video_path)

    # Rectangular container box for video display and live metrics
    video_box = st.container(border=True)
    with video_box:
        video_display = st.empty()
        status_display = st.empty()

    # Frame processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame through detection, tracking, line-crossing counter, and rendering
        annotated_frame, stats = pipeline.process(frame, conf_thresh=0.25)

        # Scale frame for display so it fits nicely on screen in a clean rectangular box
        disp_h, disp_w = annotated_frame.shape[:2]
        max_h, max_w = 480, 640
        scale = min(max_w / disp_w, max_h / disp_h, 1.0)
        if scale < 1.0:
            disp_frame = cv2.resize(annotated_frame, (int(disp_w * scale), int(disp_h * scale)), interpolation=cv2.INTER_AREA)
        else:
            disp_frame = annotated_frame

        # Render rescaled frame and live statistics
        video_display.image(cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB))
        status_display.markdown(
            f"📥 **IN: {stats['in']}** | 📤 **OUT: {stats['out']}** | 🚪 **INSIDE: {stats['inside']}** | "
            f"👥 **Total Unique: {stats['total']}** | ⚡ **FPS: {stats['fps']:.1f}**"
        )

    cap.release()
    st.success(
        f"Processing Complete! Final Counts ➔ IN: **{stats['in']}** | OUT: **{stats['out']}** | "
        f"CURRENTLY INSIDE: **{stats['inside']}** | Total Unique People: **{pipeline.total_unique_count}**"
    )
