import tempfile, cv2, streamlit as st
from pipeline import TrackingPipeline

st.title("Person Tracker")

# 1. Upload Video (Only element on the page)
video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])

# 2. Output (Automatically starts when file is uploaded)
if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_file.getbuffer())
        video_path = f.name

    pipeline = TrackingPipeline("models/best.pt")
    cap = cv2.VideoCapture(video_path)
    
    video_display = st.empty()
    status_display = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, stats = pipeline.process(frame)
        video_display.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        status_display.write(f"Active: **{stats['active']}** | Total Unique: **{stats['total']}** | FPS: **{stats['fps']:.1f}**")

    cap.release()
    st.success(f"Complete! Total unique people detected: {pipeline.total_unique_count}")
