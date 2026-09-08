import tempfile, cv2, streamlit as st
from pipeline import TrackingPipeline

st.title("Person Tracker & IN/OUT Counter")

# This code is used to configure counting line position for person IN/OUT counting in the Streamlit UI
line_pos = st.sidebar.slider("Counting Line Position Ratio", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

# 1. Upload Video
video_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])

# 2. Output (Automatically starts when file is uploaded)
if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
        f.write(video_file.getbuffer())
        video_path = f.name

    # This code is used to initialize the pipeline and set line position for person IN/OUT counting
    pipeline = TrackingPipeline("models/best.pt", line_position=line_pos)
    if hasattr(pipeline, 'set_line_position'):
        pipeline.set_line_position(line_pos)
    cap = cv2.VideoCapture(video_path)


    
    video_display = st.empty()
    status_display = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, stats = pipeline.process(frame)
        video_display.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        # This code is used to display live IN, OUT, and INSIDE statistics in the Streamlit UI for person IN/OUT counting
        status_display.write(
            f"📥 **IN: {stats['in']}** | 📤 **OUT: {stats['out']}** | 🚪 **INSIDE: {stats['inside']}** | "
            f"Active: **{stats['active']}** | Total Unique: **{stats['total']}** | FPS: **{stats['fps']:.1f}**"
        )

    cap.release()
    st.success(
        f"Processing Complete! Final Counts ➔ IN: **{stats['in']}** | OUT: **{stats['out']}** | "
        f"CURRENTLY INSIDE: **{stats['inside']}** | Total Unique People: **{pipeline.total_unique_count}**"
    )

